--- Supplementary utilities for signal operations.

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

-- Modules --
local fft_utils = require("dft_ops.utils")
local real_fft = require("dft_ops.real_fft")

-- Exports --
local M = {}

--- DOCMEMORE
-- Helper to compute a dimension length and associated power-of-2
function M.LenPower (n1, n2)
	local len, n = n1 + n2 - 1, 1

	while n < len do
		n = 2 * n
	end

	return len, n
end

--- DOCMEORE
-- Common 1D precomputations
function M.PrecomputeKernel1D (out, n, kernel, kn)
	fft_utils.PrepareRealFFT_1D(out, n, kernel, kn)
	real_fft.RealFFT_1D(out, n)

	out.n = kn
end

--- DOCMEMORE
-- Precomputed kernel method
function M.MakePrecomputedKernelFunc1D (out)
	return function(n, signal, sn, kernel)
		fft_utils.PrepareRealFFT_1D(out, n, signal, sn)
		real_fft.RealFFT_1D(out, n)
		fft_utils.Multiply_1D(out, kernel, n)
	end
end

-- Export the module.
return M