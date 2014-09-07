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
local max = math.max
local min = math.min

-- Modules --
local linear_convolution = require("signal_ops.linear_convolution")
local svd = require("linear_algebra_ops.svd")

-- Imports --
local Convolve_1D = linear_convolution.Convolve_1D

-- Exports --
local M = {}

-- --
local Columns, ColVector, RowVector = {}, {}, {}

-- --
local Opts1, Opts2 = {}, { into = {} }

--
local function AuxConvolve (out, signal, scols, sn, u, v, len)
	--
	for i = 1, scols do
		local j = 1

		for ci = i, sn, scols do
			ColVector[j], j = signal[ci], j + 1
		end

		Opts1.into = Columns[i]

		Convolve_1D(ColVector, u, Opts1)
	end

	--
	local index = 1

	for i = 1, len do
		for j = 1, scols do
			RowVector[j] = Columns[j][i]
		end

		local row = Convolve_1D(RowVector, v, Opts2)

		for k = 1, #row do
			out[index], index = row[k], index + 1
		end
	end
end

-- Intermediate convolution target, once accumulation kicks off --
local ToAdd = {}

--- DOCME
function M.Convolve_2D (signal, scols, decomp, opts)
	local csignal, sn, p = opts and opts.into or {}, opts and opts.sn or #signal, decomp.p
	local max_rank, srows, len = opts and opts.max_rank or decomp.max_rank, sn / scols, scols + p - 1
	local n = len * (srows + decomp.q - 1)

	-- Typical case: matrix was successfully decomposed.
	if max_rank > 0 then
		local u, v = decomp.uarr, decomp.varr

		-- Trim work vectors as necessary.
		for i = #ColVector, srows + 1, -1 do
			ColVector[i] = nil
		end

		for i = #RowVector, scols + 1, -1 do
			RowVector[i] = nil
		end

		--
		for i = #Columns + 1, scols do
			Columns[i] = {}
		end

		-- Ideally, the matrix was separable, in which case one set of convolutions suffices. In
		-- any case, at this point the output signal and accumulator are the same thing.
		AuxConvolve(csignal, signal, scols, sn, u[1], v[1], len)

		-- If the matrix was non-separable, refine the approximation with as many convolutions as
		-- the user allows, or until rank is reached, when the refinement is almost exact.
		for i = 2, min(max_rank, decomp.max_rank) do
			AuxConvolve(ToAdd, signal, scols, sn, u[i], v[i], len)

			for j = 1, n do
				csignal[j] = csignal[j] + ToAdd[j]
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

--- DOCME
function M.DecomposeKernel (kernel, kcols, opts)
	local kn = opts and opts.kn or #kernel
	local decomp = opts and opts.into or {}
	local uarr, varr = decomp.uarr or {}, decomp.varr or {}

	if kcols^2 == kn then
		local u, s, v = svd.SVD_Square(kernel, kcols)

		for i = 1, kcols do
			local uv, vv, j = uarr[i] or {}, varr[i] or {}, 1

			for ci = i, kn, kcols do
				uv[j], vv[j], j = u[ci], v[ci], j + 1
			end

			for k = max(#uv, #vv), j, -1 do
				uv[k], vv[k] = nil
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