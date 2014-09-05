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

-- Modules --
local svd = require("linear_algebra_ops.svd")

-- Exports --
local M = {}

--- DOCME
function M.DecomposeKernel (kernel, kcols, kn)
	kn = kn or #kernel

	if kcols^2 == kn then
		local u, s, v = svd.SVD_Square(kernel, kcols)
	else
		-- ...
	end
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