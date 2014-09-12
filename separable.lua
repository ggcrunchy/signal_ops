--- Utilities for separable kernels.

--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
-- CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
-- TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
-- SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--
-- [ MIT license: http://www.opensource.org/licenses/mit-license.php ]
--

-- Standard library imports --
local assert = assert
local max = math.max
local min = math.min

-- Modules --
local fft_utils = require("dft_ops.utils")
local linear_convolution = require("signal_ops.linear_convolution")
local real_fft = require("dft_ops.real_fft")
local svd = require("linear_algebra_ops.svd")
local utils = require("signal_ops.utils")

-- Imports --
local Convolve_1D = linear_convolution.Convolve_1D

-- Exports --
local M = {}

--
local function LoadColumn (from, to, col, size, count)
	local index = 1

	for ci = col, size, count do
		to[index], index = from[ci], index + 1
	end
end

-- --
local Columns, ColVector, RowVector = {}, {}, {}

-- --
local ColOpts, RowOpts = { into = Columns }, {}

--
local function AuxConvolve (from, count, size, u, v, len)
	local offset, signal, kernel, opts = 0, ColVector, u, ColOpts

	for _ = 1, 2 do
		for i = 1, count do
			LoadColumn(from, signal, i, size, count)

			opts.offset, offset = offset, offset + len

			Convolve_1D(signal, kernel, opts)
		end

		count, from, size, offset, signal, opts, kernel = len, Columns, offset, 0, RowVector, RowOpts, v
	end
end

-- Intermediate convolution target, once accumulation kicks off --
local ToAdd = {}

-- --
local B, C, D = {}, {}, {}

-- --
local PrecomputedKernelFunc = utils.MakePrecomputedKernelFunc_1D(C)

--
local function AuxConvolve_FFT (csignal, max_rank, decomp, scols, srows)
	local into, up, vp = csignal, decomp.up, decomp.vp
	local halfup, ulen, uarr = .5 * up, decomp.ulen, decomp.uarr
	local halfvp, vlen, varr = .5 * vp, decomp.vlen, decomp.varr
	local n, up2 = ulen * vlen, 2 * up
	local up_to = scols * up2 - 1

	for rank = 1, max_rank do
		--
		local size, u = 0, uarr[rank]

		for offset = 0, up_to, up2 do
			for j = 1, up2 do
				C[j] = Columns[offset + j]
			end

			fft_utils.Multiply_1D(C, u, up, B)

			-- ...transform back to the time domain...
			real_fft.RealIFFT_1D(B, halfup)

			-- ...and get the requested part of the result.
			for j = 1, ulen do
				D[size + 1], size = B[j], size + 1
			end
		end

		--
		local index, v = 1, varr[rank]

		for i = 1, ulen do
			LoadColumn(D, B, i, size, ulen)
			PrecomputedKernelFunc(vp, B, srows, v)

			real_fft.RealIFFT_1D(C, halfvp)

			for j = 1, vlen do
				into[index], index = C[j], index + 1
			end
		end

		--
		if rank > 1 then
			for j = 1, n do
				csignal[j] = csignal[j] + ToAdd[j]
			end
		else
			into = ToAdd
		end
	end
end

--- DOCME
function M.Convolve_2D (signal, scols, decomp, opts)
	local csignal, sn, p = opts and opts.into or {}, opts and opts.sn or #signal, decomp.p
	local max_rank, srows, len = opts and opts.max_rank or decomp.max_rank, sn / scols, scols + p - 1
	local n = len * (srows + decomp.q - 1)

	-- Typical case: matrix was successfully decomposed.
	if max_rank > 0 then

		--
		if decomp.up then
			local index, up2 = 1, 2 * decomp.up

			for i = 1, scols do
				LoadColumn(signal, B, i, sn, scols)

				utils.PrecomputeKernel_1D(C, decomp.up, B, srows)

				for i = 1, up2 do
					Columns[index], index = C[i], index + 1
				end
			end

			AuxConvolve_FFT(csignal, max_rank, decomp, scols, srows)
		else
			local u, v = decomp.uarr, decomp.varr

			-- Trim work vectors as necessary.
			for i = #ColVector, srows + 1, -1 do
				ColVector[i] = nil
			end

			for i = #RowVector, scols + 1, -1 do
				RowVector[i] = nil
			end

			-- Ideally, the matrix was separable, in which case one set of convolutions suffices. In
			-- any case, at this point the output signal and accumulator are the same thing.
			RowOpts.into = csignal

			for rank = 1, min(max_rank, decomp.max_rank) do
				AuxConvolve(signal, scols, sn, u[rank], v[rank], len)

				-- If the matrix was non-separable, refine the approximation with as many convolutions as
				-- the user allows, or until rank is reached, when the refinement is almost exact.
				if rank > 1 then
					for j = 1, n do
						csignal[j] = csignal[j] + ToAdd[j]
					end
				else
					RowOpts.into = ToAdd
				end
			end
		end

	-- Degenerate matrix: all zeroes.
	else
		for i = 1, n do
			csignal[i] = 0
		end
	end

	return csignal
end

--
local function FindRank (s, n)
	for i = 1, n do
		if s[i] < 1e-6 then
			return i - 1
		end
	end

	return n
end

-- --
local TempU, TempV

--- DOCME
function M.DecomposeKernel (kernel, kcols, opts)
	local kn, scols, srows, up, vp = opts and opts.kn or #kernel, opts and opts.scols, opts and opts.srows

	assert(not scols == not srows, "Missing signal columns or row count")

	local decomp = opts and opts.into or {}
	local uarr, varr = decomp.uarr or {}, decomp.varr or {}

	--
	if scols then
		TempU, decomp.ulen, up = TempU or {}, utils.LenPower(scols, kcols)
		TempV, decomp.vlen, vp = TempV or {}, utils.LenPower(srows, kn / kcols)

		decomp.up, decomp.vp = up, vp
	end

	--
	if kcols^2 == kn then
		local u, s, v = svd.SVD_Square(kernel, kcols)

		for i = 1, kcols do
			local uv, vv, j = uarr[i] or {}, varr[i] or {}, 1

			--
			if scols then
				TempU, TempV, uv, vv = uv, vv, TempU, TempV
			else
				uv.n, vv.n = nil
			end

			--
			for ci = i, kn, kcols do
				uv[j], vv[j], j = u[ci], v[ci], j + 1
			end

			for k = max(#uv, #vv), j, -1 do
				uv[k], vv[k] = nil
			end

			--
			if scols then
				TempU, TempV, uv, vv = uv, vv, TempU, TempV

				utils.PrecomputeKernel_1D(uv, up, TempU, kcols)
				utils.PrecomputeKernel_1D(vv, vp, TempV, kcols)
			end

			uarr[i], varr[i] = uv, vv
		end

		decomp.max_rank, decomp.p, decomp.q = FindRank(s, kcols), kcols, kcols
	else
		-- ...
	end

	decomp.uarr, decomp.varr = uarr, varr

	return decomp
end

-- Export the module.
return M