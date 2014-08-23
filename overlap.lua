--- Convolution based on overlap-add and overlap-save algorithms.
--
-- Overlap code may be adapted from:
--
-- Copyright (c) 2009, Luigi Rosa
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are
-- met:
--
--  * Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
--  * Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in
--    the documentation and/or other materials provided with the distribution
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
-- ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
-- LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
-- CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
-- SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
-- INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
-- CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
-- ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
-- POSSIBILITY OF SUCH DAMAGE.

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
local ceil = math.ceil
local huge = math.huge
local min = math.min

-- Modules --
local real_fft = require("dft_ops.real_fft")
local signal_utils = require("signal_ops.utils")

-- Exports --
local M = {}

-- Scratch buffers used to perform transforms --
local B, C = {}, {}

--
local PrecomputedKernelFunc = signal_utils.MakePrecomputedKernelFunc1D(B)

-- Computes a reasonable block length and pretransforms the kernel
local function TransformKernel1D (kernel, kn)
	local overlap = kn - 1
	local _, n = signal_utils.LenPower(4 * overlap, kn)

	signal_utils.PrecomputeKernel1D(C, n, kernel, kn)

	return overlap, n, .5 * n
end

--- One-dimensional linear convolution using the [overlap-add method](http://en.wikipedia.org/wiki/Overlap–add_method).
--
-- When _signal_ is much longer than _kernel_, this can significantly improve performance.
-- @array signal Real discrete signal...
-- @array kernel ...and kernel.
-- @ptable[opt] opts Convolve options.  Fields:
--
-- * **into**: If provided, this table will receive the convolution.
-- * **is_circular**: If true, circular convolution is performed instead.
-- * **sn**: If provided, this is the length of _signal_; otherwise, #_signal_.
-- @treturn array Convolution.
function M.OverlapAdd_1D (signal, kernel, opts)
	local sn, kn = (opts and opts.sn) or #signal, #kernel

	if sn < kn then
		signal, kernel, sn, kn = kernel, signal, kn, sn
	end

	-- Set up loop-invariant parts.
	local overlap, n, halfn = TransformKernel1D(kernel, kn)

	-- Begin with an all-zeroes signal.
	local csignal, nconv = opts and opts.into or {}, kn + sn - 1

	for i = 1, nconv do
		csignal[i] = 0
	end

	-- Read in and process each block.
	local blockn = n - overlap

	for pos = 1, sn, blockn do
		-- Read in the next part of the signal.
		local count = min(pos + blockn - 1, sn) - pos + 1

		for i = 0, count - 1 do
			B[i + 1] = signal[pos + i]
		end

		-- Multiply the (complex) results...
		PrecomputedKernelFunc(n, B, count, C, kn)

		-- ...transform back to the time domain...
		real_fft.RealIFFT_1D(B, halfn)

		-- ...and get the requested part of the result.
		local up_to, di = min(pos + n - 1, nconv), pos - 1

		for i = pos, up_to do
			csignal[i] = csignal[i] + B[i - di]
		end
	end

	-- If requested, find the circular convolution.
	if opts and opts.is_circular then
		for i = 1, overlap do
			csignal[i] = csignal[i] + csignal[sn + i]
		end

		for i = nconv, sn + 1, -1 do
			csignal[i] = nil
		end
	end

	return csignal
end

-- TODO: 2D...

-- Helper to wrap an array slot
local function Wrap (x, n)
	return x <= n and x or (x - 1) % n + 1
end

-- Read from a signal, using periodicity to account for out-of-range reads
local function PeriodicRead (out, n, to, signal, from, sn)
	from = Wrap(from, sn)

	-- If the signal wraps, split up the read count into how many samples to read until the
	-- end of the buffer, and how far to read from its beginning. If instead the signal is
	-- too short to be serviced by the full read, adjust its count to accommodate periodicity.
	local raw_ut = from + n - 1
	local up_to = Wrap(raw_ut, sn)
	local wrapped = up_to < from

	if wrapped then
		n = n - up_to
	elseif raw_ut ~= up_to then
		n = sn - from + 1
	end

	-- Read the (possibly later part of) the signal.
	for i = 0, n - 1 do
		out[to + i] = signal[from + i]
	end

	-- If the signal wrapped, read in that part too. Supply the new read and write indices.
	to = to + n

	if wrapped then
		for i = 0, up_to - 1 do
			out[to + i] = signal[i + 1]
		end

		return to + up_to, up_to
	else
		return to, from + n
	end
end

-- Fills the remainder of a buffer from a periodic signal
local function Fill (out, to, last, signal, from, sn)
	for i = to, last do
		if from > sn then
			from = 1
		end

		out[i], from = signal[from], from + 1
	end
end

--- One-dimensional linear convolution using the [overlap-save method](http://en.wikipedia.org/wiki/Overlap–save_method).
--
-- When _signal_ is much longer than _kernel_, this can significantly improve performance.
-- @array signal Real discrete signal...
-- @array kernel ...and kernel.
-- @ptable[opt] opts Convolve options.  Fields:
--
-- * **into**: If provided, this table will receive the convolution.
-- * **sn**: If provided, this is the length of _signal_; otherwise, #_signal_.
-- @treturn array Convolution.
function M.OverlapSave_1D (signal, kernel, opts)
	local sn, kn = (opts and opts.sn) or #signal, #kernel
	local is_periodic = not not (opts and opts.is_periodic)
	-- ^^^ Periodicity = Guess, based on http://www.scribd.com/doc/219373222/Overlap-Save-Add...
	-- Obviously, to actually support this would be a lot more logic...
	-- Moreover, it's reasonable that signal is infinite, which would imply a callback

	if sn < kn then
		signal, kernel, sn, kn = kernel, signal, kn, sn
	end

	-- Detect K * kn >= sn, etc. (might handle already...)
	-- For small sizes, do Goertzel?

	-- Set up loop-invariant parts.
	local overlap, n, halfn = TransformKernel1D(kernel, kn)

	-- The first "saved" samples are all zeroes.
	for i = 1, overlap do
		B[i] = 0
	end

	-- Read in each block, stepping slightly fewer than N samples to account for overlap.
	local csignal = opts and opts.into or {}
	local nconv, step = sn + kn - 1, n - overlap

	for pos = 1, nconv, step do
		-- Carry over a few samples from the last block.
		if pos > 1 then
			PeriodicRead(B, overlap, 1, signal, pos - overlap, sn)
		end

		-- Read in the new portion of the block. If there are fewer than L samples for the final
		-- block, pad it with zeroes and adjust the ranges.
		local count, up_to = n, pos + step - 1
		local diff, wi, ri = up_to - sn, PeriodicRead(B, n - overlap, kn, signal, pos, sn)

		if diff > 0 then
			count, up_to = n - diff, nconv

			if is_periodic then							

	
				Fill(B, wi, n, signal, Wrap(ri, sn), sn)

				count = n
			end
		end
		-- ^^^ TODO: Could it possibly spill over into one more (degenerate?) block?

		-- Multiply the (complex) results...
		PrecomputedKernelFunc(n, B, count, C, kn)

		-- ...transform back to the time domain...
		real_fft.RealIFFT_1D(B, halfn)

		-- ...and get the requested part of the result.
		local di = pos - kn

		for i = pos, up_to do
			csignal[i] = B[i - di]
		end
	end

	return csignal
end

-- --
local Flops = {
	{ 14, 58, 172, 440, 1038, 2358, 5264, 11644, 25594, 55946, 121644, 263136, 566438 },
	{ 58, 178, 470, 1134, 2586, 5738, 12574, 27382, 59378, 128274, 276054, 591806, 1263946 },
	{ 172, 470, 1170, 2730, 6098, 13330, 28858, 62186, 133602, 286242, 611498, 1302394, 2765458 },
	{ 440, 1134, 2730, 6242, 13762, 29794, 63986, 136914, 292290, 622658, 1323346, 2805490, 5932322 },
	{ 1038, 2586, 6098, 13762, 30082, 64706, 138210, 294306, 625538, 1327234, 2810530, 5938658, 12520002 },
	{ 2358, 5738, 13330, 29794, 64706, 138498, 294594, 624962, 1323778, 2799874, 5911874, 12458946, 26203266 },
	{ 5264, 12574, 28858, 63986, 138210, 294594, 624386, 1320322, 2788354, 5881346, 12386946, 26044290, 54659330 },
	{ 11644, 27382, 62186, 136914, 294306, 624962, 1320322, 2783746, 5862914, 12335106, 25918722, 54378242, 113897986 },
	{ 25594, 59378, 133602, 292290, 625538, 1323778, 2788354, 5862914, 12316674, 25851906, 54200834, 113483266, 237249538 },
	{ 55946, 128274, 286242, 622658, 1327234, 2799874, 5881346, 12335106, 25851906, 54140930, 113275906, 236715010, 493996034 },
	{ 121644, 276054, 611498, 1323346, 2810530, 5911874, 12386946, 25918722, 54200834, 113275906, 236539906, 493406210, 1027944450 },
	{ 263136, 591806, 1302394, 2805490, 5938658, 12458946, 26044290, 54378242, 113483266, 236715010, 493406210, 1027465218, 2137194498 },
	{ 566438, 1263946, 2765458, 5932322, 12520002, 26203266, 54659330, 113897986, 237249538, 493996034, 1027944450, 2137194498, 4438917122 }
}

-- --
local SRows, SCols, KRows, KCols

-- --
local LSetX, LSetY = {}, {}

--
local function FindSets (low, lset)
	local index, size = 1, 0

	while 2^(index - 1) < low do
		index = index + 1
	end

	for i = index, 13 do
		lset[size + 1], size = 2^(i - 1) - low + 1, size + 1
	end

	return size, index
end

--
local function Find (bx, by, dimx, dimy)
	local sizex, x0 = FindSets(bx, LSetX)
	local vmin, xi, yi, sizey, yc, yset = huge

	if bx == by then
		sizey, yc, yset = sizex, x0, LSetX
	else
		sizey, yc = FindSets(by, LSetY)
		yset = LSetY
	end

	for j = 1, sizey do
		local yfactor, xc, row = ceil(dimy / yset[j]), x0, Flops[yc]

		for i = 1, sizex do
			local v = ceil(dimx / LSetX[i]) * row[xc]

			if v < vmin then
				vmin, xi, yi = v, xc, yc
			end

			xc = xc + 1
		end

		yc = yc + 1
	end

	return vmin, xi, yi
end

-- --
local Nx, Ny, Lx, Ly, Swap

--
local function GetSizes (index, dim)
	local n = 2^(index - 1)

	return n, n - dim + 1
end

--- DOCME
function M.OverlapAdd_2D (signal, kernel, scols, kcols, opts)
	local sn, kn = #signal, #kernel
	local srows = sn / scols
	local krows = kn / kcols
	local dimx = scols + kcols - 1
	local dimy = srows + krows - 1

	--
	if srows ~= SRows or scols ~= SCols or krows ~= KRows or kcols ~= KCols then
		local max1, ix1, iy1 = Find(kcols, krows, dimx, dimy)
		local max2, ix2, iy2 = Find(scols, srows, dimx, dimy)

		Swap = max2 < max1

		if not Swap then
			Nx, Lx = GetSizes(ix1, kcols)
			Ny, Ly = GetSizes(iy1, krows)
		else
			Nx, Lx = GetSizes(ix2, scols)
			Ny, Ly = GetSizes(iy2, srows)
		end

		SRows, SCols = srows, scols
		KRows, KCols = krows, kcols
	end

	--
	if Swap then
		signal, kernel = kernel, signal
	end

	--
	local csignal = opts and opts.into or {}

	for i = 1, dimx * dimy do
		csignal[i] = 0
	end

--	kernel2 = fft2(kernel, Nx, Ny);

-- ???
--	signal2 = signal
--	signal2[dimx, dimy] = 0?

	for xstart = 1, dimx, Lx do
		local xend = min(xstart + Lx - 1, dimx)
		local endx = min(dimx, xstart + Nx - 1)

		for ystart = 1, dimy, Ly do
			local yend = min(ystart + Ly - 1, dimy)

			-- X = fft2(signal2[xstart : xend, ystart : yend], Nx, Ny)
			-- Y = ifft2(X .* kernel2)

			local endy = min(dimy, ystart + Ny - 1)

			-- csignal[xstart : endx, ystart : endy] = csignal[xstart : endx, ystart : endy] + Y[1 : endx - xstart + 1, 1 : endy - ystart + 1]
		end
	end
end

-- Export the module.
return M