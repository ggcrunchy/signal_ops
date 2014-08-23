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
<<<<<<< HEAD
	{ 14, 58, 172, 440, 1038, 2358, 5264, 11644, 25594, 55946, 121644, 263136, 566438 },
=======
	{ 14, 58,172, 440, 1038, 2358, 5264, 11644, 25594, 55946, 121644, 263136, 566438 },
>>>>>>> c37b21704177120532f66131d3411f4d627ffdc2
	{ 58, 178, 470, 1134, 2586, 5738, 12574, 27382, 59378, 128274, 276054, 591806, 1263946 },
	{ 172, 470, 1170, 2730, 6098, 13330, 28858, 62186, 133602, 286242, 611498, 1302394, 2765458 },
	{ 440, 1134, 2730, 6242, 13762, 29794, 63986, 136914, 292290, 622658, 1323346, 2805490, 5932322 },
	{ 1038, 2586, 6098, 13762, 30082, 64706, 138210, 294306, 625538, 1327234, 2810530, 5938658, 12520002 },
	{ 2358, 5738, 13330, 29794, 64706, 138498, 294594, 624962, 1323778, 2799874, 5911874, 12458946, 26203266 },
	{ 5264, 12574, 28858, 63986, 138210, 294594, 624386, 1320322, 2788354, 5881346, 12386946, 26044290, 54659330 },
	{ 11644, 27382, 62186, 136914, 294306, 624962, 1320322, 2783746, 5862914, 12335106, 25918722, 54378242, 113897986 },
	{ 25594, 59378, 133602, 292290, 625538, 1323778, 2788354, 5862914, 12316674, 25851906, 54200834, 113483266, 237249538 },
	{ 55946, 128274, 286242, 622658, 1327234, 2799874, 5881346, 12335106, 25851906, 54140930, 113275906, 236715010, 493996034 },
<<<<<<< HEAD
	{ 121644, 276054, 611498, 1323346, 281053, 5911874, 12386946, 25918722, 54200834, 113275906, 236539906, 49340621, 1027944450 },
	{ 263136, 591806, 1302394, 2805490, 5938658, 12458946, 2604429, 54378242, 113483266, 236715010, 493406210, 1027465218, 2137194498 },
	{ 566438, 1263946, 2765458, 5932322, 12520002, 26203266, 5465933, 113897986, 237249538, 493996034, 1027944450, 2137194498, 4438917122 }
}

-- --
local SRows, SCols, KRows, KCols, XMax, YMax
=======
	{ 0.00012164400000, 0.00027605400000, 0.00061149800000, 0.00132334600000, 0.00281053000000, 0.00591187400000, 0.01238694600000,
	  0.02591872200000, 0.05420083400000, 0.11327590600000, 0.23653990600000, 0.49340621000000, 1.02794445000000 },
	{ 0.00026313600000, 0.00059180600000, 0.00130239400000, 0.00280549000000, 0.00593865800000, 0.01245894600000, 0.02604429000000,
	  0.05437824200000,  0.11348326600000, 0.23671501000000, 0.49340621000000, 1.02746521800000, 2.13719449800000 },
	{ 0.00056643800000, 0.00126394600000, 0.00276545800000, 0.00593232200000, 0.01252000200000, 0.02620326600000, 0.05465933000000,
	  0.11389798600000,  0.23724953800000, 0.49399603400000, 1.02794445000000, 2.13719449800000, 4.43891712200000 }
}

for i = 11, 13 do
	local arr = Flops[i]

	for j = 1, #arr do
		arr[j] = 1e9 * arr[j]
	end
end

-- --
local DimX, DimY, XMax, YMax
>>>>>>> c37b21704177120532f66131d3411f4d627ffdc2

-- --
local Lx, Ly = {}, {}

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
-- nx=1:13;
-- validsetx=find(2.^(nx-1)>bx-1);
-- nx=nx(validsetx);
-- Lx=2.^(nx-1)-bx+1;
-- sizex=length(nx);
end

--
local function Find (bx, by, dimx, dimy)
	local sizex, x = FindSets(bx, Lx)
	local sizey, y = FindSets(by, Ly)
	local vmin, xpos, ypos = huge

	for j = 1, sizey do
		local yfactor, xcur, row = ceil(dimy / Ly[j]), x, Flops[y]

		for i = 1, sizex do
			local curv = ceil(dimx / Lx[i]) * row[xcur]

			if curv < vmin then
				curv, xpos, ypos = vmin, xcur, y
			end

			xcur = xcur + 1
		end

		y = y + 1
	end

	return vmin, xpos, ypos
	--[[
    matrice=zeros(sizex,sizey);
    for ii=1:sizex
        for jj=1:sizey
            matrice(ii,jj)=ceil(dimx/Lx(ii))*ceil(dimy/Ly(jj))*fftflops(nx(ii),ny(jj));
        end
    end
    [massimo_vettore,posizione_vettore]=min(matrice);
    [massimo,posizione]=min(massimo_vettore);
    y_max=posizione;
    x_max=posizione_vettore(posizione);
    massimo;
	]]
<<<<<<< HEAD
	--[[
		If A is a matrix, min(A) treats the columns of A as vectors, returning a row vector containing the minimum element from each column.

		[C,I] = min(...) finds the indices of the minimum values of A, and returns them in output vector I. If there are several identical
		minimum values, the index of the first one found is returned.
	]]
=======
>>>>>>> c37b21704177120532f66131d3411f4d627ffdc2
end

--- DOCME
function M.OverlapAdd_2D (signal, kernel, scols, kcols, opts)
	local sn, kn = #signal, #kernel
	local srows = sn / scols
	local krows = kn / kcols
	local dimx = scols + kcols - 1
	local dimy = srows + krows - 1

	--
<<<<<<< HEAD
	if srows ~= SRows or scols ~= SCols or krows ~= KRows or kcols ~= KCols then
=======
	if dimx ~= DimX or dimy ~= DimY then
>>>>>>> c37b21704177120532f66131d3411f4d627ffdc2
		local max1, xmax1, ymax1 = Find(srows, krows, dimx, dimy)
		local max2, xmax2, ymax2 = Find(scols, kcols, dimx, dimy)

		if max1 < max2 then
			XMax, YMax = xmax1, ymax1
		else
			XMax, YMax = xmax2, ymax2
		end

<<<<<<< HEAD
		SRows, SCols = srows, scols
		KRows, KCols = krows, kcols
=======
		DimX, DimY = dimx, dimy
>>>>>>> c37b21704177120532f66131d3411f4d627ffdc2
	end
--[[
    nx=1:13;
    ny=1:13;
    validsetx=find(2.^(nx-1)>bx-1);
    validsety=find(2.^(ny-1)>by-1);
    nx=nx(validsetx);
    ny=ny(validsety);
    Lx=2.^(nx-1)-bx+1;
    Ly=2.^(ny-1)-by+1;
    sizex=length(nx);
    sizey=length(ny);
    matrice=zeros(sizex,sizey);
    for ii=1:sizex
        for jj=1:sizey
            matrice(ii,jj)=ceil(dimx/Lx(ii))*ceil(dimy/Ly(jj))*fftflops(nx(ii),ny(jj));
        end
    end
    [massimo_vettore,posizione_vettore]=min(matrice);
    [massimo,posizione]=min(massimo_vettore);
    y_max=posizione;
    x_max=posizione_vettore(posizione);
    massimo;
    %.......................................
    nx2=1:13;
    ny2=1:13;
    validsetx2=find(2.^(nx2-1)>ax-1);
    validsety2=find(2.^(ny2-1)>ay-1);
    nx2=nx2(validsetx2);
    ny2=ny2(validsety2);
    Lx2=2.^(nx2-1)-ax+1;
    Ly2=2.^(ny2-1)-ay+1;
    sizex2=length(nx2);
    sizey2=length(ny2);
    matrice2=zeros(sizex2,sizey2);
    for ii=1:sizex2
        for jj=1:sizey2
            matrice2(ii,jj)=ceil(dimx/Lx2(ii))*ceil(dimy/Ly2(jj))*fftflops(nx2(ii),ny2(jj));
        end
    end
    [massimo_vettore2,posizione_vettore2]=min(matrice2);
    [massimo2,posizione2]=min(massimo_vettore2);
    y_max2=posizione2;
    x_max2=posizione_vettore2(posizione2);
    massimo2;]]
end

--[[
[ax,ay]=size(a);
[bx,by]=size(b);
dimx=ax+bx-1;
dimy=ay+by-1;

if (nargin<3)||(mode==0) 
    % figure out which nfftx, nffty, Lx and Ly to use
    %--------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fftflops=zeros(13);
    fftflops(1:10,:)=[14          58         172         440        1038        2358     5264       11644       25594       55946      121644      263136     566438;...
            58         178         470        1134        2586        5738     12574       27382       59378      128274      276054      591806    1263946;...
            172         470        1170        2730        6098       13330    28858       62186      133602      286242      611498     1302394    2765458;...
            440        1134        2730        6242       13762       29794    63986      136914      292290      622658     1323346     2805490    5932322;...
            1038        2586        6098       13762       30082       64706   138210      294306      625538     1327234     2810530     5938658   12520002;...
            2358        5738       13330       29794       64706      138498   294594      624962     1323778     2799874     5911874    12458946   26203266;...
            5264       12574       28858       63986      138210      294594   624386     1320322     2788354     5881346    12386946    26044290   54659330;...
            11644       27382       62186      136914      294306      624962  1320322     2783746     5862914    12335106    25918722    54378242  113897986;...
            25594       59378      133602      292290      625538     1323778  2788354     5862914    12316674    25851906    54200834   113483266  237249538;...
            55946      128274      286242      622658     1327234     2799874  5881346    12335106    25851906    54140930   113275906   236715010  493996034];
    fftflops(11:13,:)=1.0e+009 *[0.00012164400000   0.00027605400000   0.00061149800000   0.00132334600000     0.00281053000000   0.00591187400000   0.01238694600000   0.02591872200000    0.05420083400000   0.11327590600000   0.23653990600000   0.49340621000000    1.02794445000000;...
            0.00026313600000   0.00059180600000   0.00130239400000   0.00280549000000     0.00593865800000   0.01245894600000   0.02604429000000   0.05437824200000    0.11348326600000   0.23671501000000   0.49340621000000   1.02746521800000    2.13719449800000;...
            0.00056643800000   0.00126394600000   0.00276545800000   0.00593232200000     0.01252000200000   0.02620326600000   0.05465933000000   0.11389798600000    0.23724953800000   0.49399603400000   1.02794445000000   2.13719449800000    4.43891712200000]; 
    
    
    
    
    %.....................................
    nx=1:13;
    ny=1:13;
    validsetx=find(2.^(nx-1)>bx-1);
    validsety=find(2.^(ny-1)>by-1);
    nx=nx(validsetx);
    ny=ny(validsety);
    Lx=2.^(nx-1)-bx+1;
    Ly=2.^(ny-1)-by+1;
    sizex=length(nx);
    sizey=length(ny);
    matrice=zeros(sizex,sizey);
    for ii=1:sizex
        for jj=1:sizey
            matrice(ii,jj)=ceil(dimx/Lx(ii))*ceil(dimy/Ly(jj))*fftflops(nx(ii),ny(jj));
        end
    end
    [massimo_vettore,posizione_vettore]=min(matrice);
    [massimo,posizione]=min(massimo_vettore);
    y_max=posizione;
    x_max=posizione_vettore(posizione);
    massimo;
    %.......................................
    nx2=1:13;
    ny2=1:13;
    validsetx2=find(2.^(nx2-1)>ax-1);
    validsety2=find(2.^(ny2-1)>ay-1);
    nx2=nx2(validsetx2);
    ny2=ny2(validsety2);
    Lx2=2.^(nx2-1)-ax+1;
    Ly2=2.^(ny2-1)-ay+1;
    sizex2=length(nx2);
    sizey2=length(ny2);
    matrice2=zeros(sizex2,sizey2);
    for ii=1:sizex2
        for jj=1:sizey2
            matrice2(ii,jj)=ceil(dimx/Lx2(ii))*ceil(dimy/Ly2(jj))*fftflops(nx2(ii),ny2(jj));
        end
    end
    [massimo_vettore2,posizione_vettore2]=min(matrice2);
    [massimo2,posizione2]=min(massimo_vettore2);
    y_max2=posizione2;
    x_max2=posizione_vettore2(posizione2);
    massimo2;
    %.......................................
    if massimo<massimo2
        nfftx=2^(nx(x_max)-1);
        nffty=2^(ny(y_max)-1);
        
        Lx=nfftx-bx+1;
        Ly=nffty-by+1;
        %--------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        B=fft2(b,nfftx,nffty);
        out=zeros(dimx,dimy);
        a2=a;
        a2(dimx,dimy)=0;
        
        xstart=1;
        while xstart <= dimx
            xend=min(xstart+Lx-1,dimx);
            ystart=1;
            while ystart <= dimy
                yend=min(ystart+Ly-1,dimy);
                %---------------------
                X=fft2(a2(xstart:xend,ystart:yend),nfftx,nffty);
                Y=ifft2(X.*B);
                endx=min(dimx,xstart+nfftx-1);
                endy=min(dimy,ystart+nffty-1);
                out(xstart:endx,ystart:endy)=out(xstart:endx,ystart:endy)+Y(1:(endx-xstart+1),1:(endy-ystart+1));
                ystart=ystart+Ly;
                %---------------------
            end
            xstart=xstart+Lx;
        end
        
        if ~(any(any(imag(a)))||any(any(imag(b))))
            out=real(out);
        end
        return;
    else
        nfftx=2^(nx2(x_max2)-1);
        nffty=2^(ny2(y_max2)-1);
        
        Lx2=nfftx-ax+1;
        Ly2=nffty-ay+1;
        %--------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        A=fft2(a,nfftx,nffty);
        out=zeros(dimx,dimy);
        b2=b;
        b2(dimx,dimy)=0;
        
        xstart=1;
        while xstart <= dimx
            xend=min(xstart+Lx2-1,dimx);
            ystart=1;
            while ystart <= dimy
                yend=min(ystart+Ly2-1,dimy);
                %---------------------
                X=fft2(b2(xstart:xend,ystart:yend),nfftx,nffty);
                Y=ifft2(X.*A);
                endx=min(dimx,xstart+nfftx-1);
                endy=min(dimy,ystart+nffty-1);
                out(xstart:endx,ystart:endy)=out(xstart:endx,ystart:endy)+Y(1:(endx-xstart+1),1:(endy-ystart+1));
                ystart=ystart+Ly2;
                %---------------------
            end
            xstart=xstart+Lx2;
        end
        
        if ~(any(any(imag(a)))||any(any(imag(b))))
            out=real(out);
        end
        return;
    end
]]

-- Export the module.
return M