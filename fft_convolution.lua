--- Fast Fourier Transform-based convolution operations.

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
local fft = require("dft_ops.fft")
local fft_utils = require("dft_ops.utils")
local goertzel = require("dft_ops.goertzel")
local real_fft = require("dft_ops.real_fft")
local signal_utils = require("signal_ops.utils")
local two_ffts = require("dft_ops.two_ffts")

-- Imports --
local LenPower = signal_utils.LenPower

-- Exports --
local M = {}

-- Scratch buffers used to perform transforms --
local B, C = {}, {}

-- One-dimensional FFT-based convolution methods --
local AuxMethod1D = {}

-- Goertzel method
function AuxMethod1D.goertzel (n, signal, sn, kernel, kn)
	fft_utils.PrepareTwoGoertzels_1D(B, C, n, signal, sn, kernel, kn)
	goertzel.TwoGoertzelsThenMultiply_1D(B, C, n)
end

-- Precomputed kernel method
AuxMethod1D.precomputed_kernel = signal_utils.MakePrecomputedKernelFunc1D(B)

-- Separate FFT's method
function AuxMethod1D.separate (n, signal, sn, kernel, kn)
	fft_utils.PrepareSeparateFFTs_1D(B, C, n, signal, sn, kernel, kn)
	fft.FFT_1D(B, n)
	fft.FFT_1D(C, n)
	fft_utils.Multiply_1D(B, C, n)
end

-- Two FFT's method
function AuxMethod1D.two_ffts (n, signal, sn, kernel, kn)
	fft_utils.PrepareTwoFFTs_1D(B, n, signal, sn, kernel, kn)
	two_ffts.TwoFFTsThenMultiply_1D(B, n)
end

-- Default one-dimensional FFT-based convolution method
local DefMethod1D = AuxMethod1D.two_ffts

-- Performs the steps of a 1D FFT-based convolve
local function DoFFT_1D (out, clen, method, signal, sn, kernel, kn, halfn, n)
	-- Multiply the (complex) results...
	method(n, signal, sn, kernel, kn)

	-- ...transform back to the time domain...
	real_fft.RealIFFT_1D(B, halfn)

	-- ...and get the requested part of the result.
	for i = 1, clen do
		out[i] = B[i]
	end
end

-- Helper to kernel size, taking method into account
local function KernelSize (kernel, method)
	return method ~= "precomputed_kernel" and #kernel or kernel.n
end

--- One-dimensional linear convolution using fast Fourier transforms. For certain _signal_
-- and _kernel_ combinations, this may be significantly faster than @{signal_ops.linear_convolution.Convolve_1D}.
-- @array signal Real discrete signal...
-- @array kernel ...and kernel.
-- @ptable[opt] opts Convolve options. Fields:
--
-- * **into**: If provided, this table will receive the convolution.
-- * **method**: If this is **"goertzel"**, the transforms are done using the [Goertzel algorithm](http://en.wikipedia.org/wiki/Goertzel_algorithm),
-- which may offer better performance in some cases. If it is **"precomputed_kernel"**, the
-- _kernel_ argument is assumed to already be FFT'd. If it is **"separate"**, two FFT's are
-- computed separately. Otherwise, the two real FFT's are computed as one complex FFT.
-- @treturn array Convolution.
-- @see PrecomputeKernel_1D
function M.Convolve_1D (signal, kernel, opts)
	-- Determine how much padding is needed to have matching power-of-2 sizes.
	local method = opts and opts.method
	local sn, kn = #signal, KernelSize(kernel, method)
	local clen, n = LenPower(sn, kn)

	-- Perform some variant of IFFT(FFT(signal) * FFT(kernel)).
	local csignal = opts and opts.into or {}

	DoFFT_1D(csignal, clen, AuxMethod1D[method] or DefMethod1D, signal, sn, kernel, kn, .5 * n, n)

	return csignal
end

-- Two-dimensional FFT-based convolution methods --
local AuxMethod2D = {}

-- Goertzel method
function AuxMethod2D.goertzel (m, n, signal, scols, kernel, kcols, sn, kn)
	fft_utils.PrepareTwoGoertzels_2D(B, C, m, n, signal, scols, kernel, kcols, sn, kn)
	goertzel.TwoGoertzelsThenMultiply_2D(B, C, m, n)
end

-- Precomputed kernel method
function AuxMethod2D.precomputed_kernel (m, n, signal, scols, kernel, _, sn, _, area)
	fft_utils.PrepareRealFFT_2D(B, area, signal, scols, m, sn)
	real_fft.RealFFT_2D(B, m, n)
	fft_utils.Multiply_2D(B, kernel, m, n)
end

-- Separate FFT's method
function AuxMethod2D.separate (m, n, signal, scols, kernel, kcols, sn, kn)
	fft_utils.PrepareSeparateFFTs_2D(B, C, m, n, signal, scols, kernel, kcols, sn, kn)
	fft.FFT_2D(B, m, n)
	fft.FFT_2D(C, m, n)
	fft_utils.Multiply_2D(B, C, m, n)
end

-- Two FFT's method
function AuxMethod2D.two_ffts (m, n, signal, scols, kernel, kcols, sn, kn, area)
	fft_utils.PrepareTwoFFTs_2D(B, area, signal, scols, kernel, kcols, m, sn, kn)
	two_ffts.TwoFFTsThenMultiply_2D(B, m, n)
end

-- Default two-dimensional FFT-based convolution method
local DefMethod2D = AuxMethod2D.two_ffts

-- Performs the steps of a 2D FFT-based convolve
local function DoFFT_2D (out, method, signal, scols, sn, kernel, kcols, kn, halfm, m, n, area, w, h)
	-- Multiply the (complex) results...
	method(m, n, signal, scols, kernel, kcols, sn, kn, area)

	-- ...transform back to the time domain...
	real_fft.RealIFFT_2D(B, halfm, n)

	-- ...and get the requested part of the result.
	local offset, index = 0, 1

	for _ = 1, h do
		for j = 1, w do
			out[index], index = B[offset + j], index + 1
		end

		offset = offset + m
	end
end

--- Two-dimensional linear convolution using fast Fourier transforms. For certain _signal_
-- and _kernel_ combinations, this may be significantly faster than @{signal_ops.linear_convolution.Convolve_2D}.
-- @array signal Real discrete signal...
-- @array kernel ...and kernel.
-- @uint scols Number of columns in _signal_... 
-- @uint kcols ... and in _kernel_.
-- @ptable[opt] opts Convolve options. Fields:
--
-- * **into**: If provided, this table will receive the convolution.
-- * **method**: As per @{Convolve_1D}, but with 2D variants.
-- @treturn array Convolution.
-- @treturn uint Number of columns in the convolution. Currently, only the **"full"** shape
-- is supported, i.e. _scols_ + _kcols_ - 1.
-- @see PrecomputeKernel_2D
function M.Convolve_2D (signal, kernel, scols, kcols, opts)
	-- Determine how much padding each dimension needs, to have matching power-of-2 sizes.
	local method = opts and opts.method
	local sn, kn = #signal, KernelSize(kernel, method)
	local srows = sn / scols
	local krows = kn / kcols
	local w, m = LenPower(scols, kcols)
	local h, n = LenPower(srows, krows)

	-- Perform some variant of IFFT(FFT(signal) * FFT(kernel)).
	local csignal = opts and opts.into or {}

	DoFFT_2D(csignal, AuxMethod2D[method] or DefMethod2D, signal, scols, sn, kernel, kcols, kn, .5 * m, m, n, m * n, w, h)

	return csignal
end

--- Precomputes a kernel, e.g. for consumption by the **"precomputed_kernel"** option of
-- @{Convolve_1D}.
-- @array out Computed kernel. Assumed to be distinct from _kernel_.
-- @uint sn Size of corresponding real discrete signal.
-- @array kernel Real discrete kernel.
function M.PrecomputeKernel_1D (out, sn, kernel)
	local kn = #kernel
	local _, n = LenPower(sn, kn)

	signal_utils.PrecomputeKernel1D(out, n, kernel, kn)
end

--- Precomputes a kernel, e.g. for consumption by the **"precomputed_kernel"** option of
-- @{Convolve_2D}.
-- @array out Computed kernel. Assumed to be distinct from _kernel_.
-- @uint sn Size of corresponding real discrete signal.
-- @array kernel Real discrete kernel.
-- @uint scols Number of columns in signal... 
-- @uint kcols ... and in _kernel_.
function M.PrecomputeKernel_2D (out, sn, kernel, scols, kcols)
	local kn = #kernel
	local srows = sn / scols
	local krows = kn / kcols
	local _, m = LenPower(scols, kcols)
	local _, n = LenPower(srows, krows)
	local area = m * n

	fft_utils.PrepareRealFFT_2D(out, area, kernel, kcols, m, kn)
	real_fft.RealFFT_2D(out, m, n)

	out.n = kn
end

--- Enters a loop, fetching a new signal on each iteration.
--
-- Each pair of this variable signal and the constant kernel are then convolved, in much
-- the same way they would be by @{Convolve_1D}.
-- @uint sn Size of real discrete signal.
-- @array kernel Real discrete kernel.
-- @callable func Called as
--    signal, out = func(sn, arg)
-- where _signal_ is a real discrete signal of size _sn_ (additional values beyond this will
-- be ignored) and _out_ will receive the convolution.
--
-- The loop aborts when _signal_ is **nil**.
-- @ptable[opt] opts Convolve options. Fields:
--
-- * **arg**: If provided, this will be the argument to _func_.
-- * **method**: As per @{Convolve_1D}.
function M.SerialConvolve_1D (sn, kernel, func, opts)
	-- Determine how much padding is needed to have matching power-of-2 sizes.
	local method = opts and opts.method
	local kn = KernelSize(kernel, method)
	local clen, n = LenPower(sn, kn)

	-- Perform some variant of IFFT(FFT(signal) * FFT(kernel)) in a loop, where a new signal
	-- and output vector are polled each iteration.
	method = AuxMethod1D[method] or DefMethod1D

	local arg, halfn = opts and opts.arg, .5 * n

	repeat
		local signal, out = func(sn, arg)

		if signal then
			DoFFT_1D(out, clen, method, signal, sn, kernel, kn, halfn, n)
		end
	until not signal
end

--- Enters a loop, fetching a new signal on each iteration.
--
-- Each pair of this variable signal and the constant kernel are then convolved, in much
-- the same way they would be by @{Convolve_2D}.
-- @uint sn Size of real discrete signal.
-- @array kernel Real discrete kernel.
-- @uint scols Number of columns in signal... 
-- @uint kcols ... and in _kernel_.
-- @callable func Called as
--    signal, out = func(sn, arg)
-- where _signal_ is a real discrete signal of size _sn_ (additional values beyond this will
-- be ignored) and _out_ will receive the convolution.
--
-- The loop aborts when _signal_ is **nil**.
-- @ptable[opt] opts Convolve options. Fields:
--
-- * **arg**: If provided, this will be the argument to _func_.
-- * **method**: As per @{Convolve_2D}.
function M.SerialConvolve_2D (sn, kernel, scols, kcols, func, opts)
	-- Determine how much padding each dimension needs, to have matching power-of-2 sizes.
	local method = opts and opts.method
	local kn = KernelSize(kernel, method)
	local srows = sn / scols
	local krows = kn / kcols
	local w, m = LenPower(scols, kcols)
	local h, n = LenPower(srows, krows)

	-- Perform some variant of IFFT(FFT(signal) * FFT(kernel)) in a loop, where a new signal
	-- and output matrix are polled each iteration.
	method = AuxMethod2D[method] or DefMethod2D

	local arg, halfm, area = opts and opts.arg, .5 * m, m * n

	repeat
		local signal, out = func(arg)

		if signal then
			DoFFT_2D(out, method, signal, scols, sn, kernel, kcols, kn, halfm, m, n, area, w, h)
		end
	until not signal
end

-- TODO: Separable filters support for 2D?

-- Export the module.
return M