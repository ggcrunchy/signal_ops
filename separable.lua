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
local Convolve_2D = linear_convolution.Convolve_2D

-- Exports --
local M = {}

-- --
local Opts1, Opts2 = { into = {}, shape = "same" }, { shape = "same" }

-- --
local ToAdd = {}

--- DOCME
function M.Convolve_2D (signal, kernel, scols, opts)
	local csignal, max_rank = opts and opts.into or {}, opts and opts.max_rank or kernel.max_rank

	if max_rank > 0 then
		Opts2.into = csignal

		--
		local n, u, v = kernel.n, kernel.uarr, kernel.varr

		Convolve_2D(Convolve_2D(signal, u[1], scols, 1, Opts1), v[1], scols, n, Opts2)

		Opts2.into = ToAdd

		for i = 2, min(max_rank, kernel.max_rank) do
			--
		end
	else
		for i = 1, #signal do
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

		decomp.max_rank, decomp.n = FindRank(s, kcols), kcols
	else
		-- ...
	end

	decomp.uarr, decomp.varr = uarr, varr

	return decomp
end

--[=[
local NN=25
local S, K1 = {}, {}--{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 1, 2, 1 }
local K2 = K1
for i = 1, NN^2 do
	S[i]=i
end
for i = 1, NN do
	K1[i]=1
end
local opts1, opts2 = { into = {}, shape = "same" }, { into = {}, shape = "same" }
for i = 1, 300 do
	aa.Convolve_2D(S, K1, NN, 1, opts1)
	aa.Convolve_2D(opts1.into, K2, NN, NN, opts2)
end
]=]

-- Export the module.
return M