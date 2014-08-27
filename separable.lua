--- Utilities for separable kernels.
--
-- Some code adapted from [Separate Kernel in 1D kernels](http://www.mathworks.com/matlabcentral/fileexchange/28218-separate-kernel-in-1d-kernels):
--
-- Copyright (c) 2010, Dirk-Jan Kroon
-- All rights reserved.
--
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

-- Exports --
local M = {}

-- 
local function Filter1DToFilterND (par, data, K1)
	if nargin==2 then
	 Kt=ones(data.sizeH)
 
	 for i = 1, data.n do
--		 p = par(data.sep_parb(i) : data.sep_pare(i))
--		 p = p(:)
		 dim = ones(1, data.n)
--		 dim(i) = data.sep_parl(i)
--		 Ki = reshape(p(:), dim)
		 dim = data.sizeH
--		 dim(i) = 1
--		 Kt = Kt .* repmat(Ki, dim)
	 end
	else
	  Kt=ones(data.sizeHreal)
  
	  for i = 1, data.n do
		dim = data.sizeHreal
--		dim(i) = 1
--		Kt = Kt .* repmat(K1{i}, dim)
	  end
	end

	return Kt
end

--
local function FilterCorrSign (par, data)
	Ert = zeros(1, length(par))
	ERR = inf
	t = 0
--	par = sign(rand(size(par)) - 0.5) .* par

	while t < ERR do
		-- Calculate the approximation of the ND kernel if using the 1D kernels.
		KN = Filter1DToFilterND(par, data)

		-- Calculate the absolute error.
	--	ERR = sum(abs(data.H(:) - KN(:)))

		-- Flip the sign of every 1D filter value, and look if the error improves.
		for i = 1, length(par) do
		--	par2 = par
		--	par2(i) = -par2(i)

			KN = Filter1DToFilterND(par2, data)

		--	Ert(i) = sum(abs(data.H(:) - KN(:)))
		end

		-- Flip the sign of the 1D filter value with the largest improvement
		t, j = min(Ert)

		if t < ERR then
--			par(j) = -par(j)
		end

		return par
	end
end

--
local function RemoveZeroRows (H)
	-- Remove whole columns / rows / planes with zeros, because we know beforehand that they
	-- will give a kernel 1D value of 0 and will otherwise increase the error in the end result.
	preserve_zeros = zeros(numel(H), 2)
	pz=0
	sizeH = size(H)

	for i = 1, ndims(H) do
--		H2D = reshape(H, size(H,1), [])
		check_zero = not any(H2D, 2)

		if any(check_zero) then
			zero_rows = find(check_zero)

			for j = 1, length(zero_rows) do
				pz = pz + 1
--				preserve_zeros(pz, :) = [i zero_rows(j)]
--				sizeH(1) = sizeH(1) - 1
			end

--			H2D(check_zero, :) = []
			H = reshape(H2D, sizeH)
		end

		H = shiftdim(H, 1)
--		sizeH = circshift(sizeH, [0 -1])
		H = reshape(H, sizeH)
	end

--	preserve_zeros = preserve_zeros(1 : pz, :)

	return H, preserve_zeros
end

--
local function InitializeDataStruct (H)
	data.sizeHreal = size(H)
	data.nreal = ndims(H)
	H, preserve_zeros = RemoveZeroRows(H)
	data.H = H
	data.n = ndims(H)
	data.preserve_zeros = preserve_zeros
--	data.H(H == 0) = eps
	data.sizeH = size(data.H)
--	data.sep_parb = cumsum([1 data.sizeH(1 : data.n - 1)])
	data.sep_pare = cumsum(data.sizeH)
	data.sep_parl = data.sep_pare - data.sep_parb + 1
--	data.par = (1 : numel(H)) + 1
	return data
end

--
local function ReaddZeroRows (data, K1)
	-- Re-add the 1D kernel values responding to a whole column / row or plane of zeros.
	for i = 1, size(data.preserve_zeros, 1) do
		di = data.preserve_zeros(i, 1)
		pos = data.preserve_zeros(i, 2)

		if di > length(K1) then
--			K1{di} = 1
		end

--		val = K1{di}
--		val = val(:)
--		val = [val(1 : pos - 1); 0; val(pos : end)]
		dim = ones(1, data.nreal)
--		dim(di) = length(val)
--		K1{di} = reshape(val, dim)
	end

	return K1
end

--
local function MakeMatrix (data)
	 M = zeros(numel(data.H), sum(data.sizeH))
--	 K1 = (1 : numel(data.H))'

	 for i = 1, data.n do
--		p = data.par(data.sep_parb(i) : data.sep_pare(i))
--		p = p(:)
		dim = ones(1, data.n)
--		dim(i) = data.sep_parl(i)
--		Ki = reshape(p(:), dim)
		dim = data.sizeH
--		dim(i) = 1
		K2 = repmat(Ki, dim) - 1
--		M(sub2ind(size(M), K1(:), K2(:))) = 1
	 end
 
	return M
end 

--
local function ValueListToFilter1D (par, data)
	 K = cell(1, data.n)
 
	 for i = 1, data.n do
--		 p = par(data.sep_parb(i) : data.sep_pare(i))
--		 p = p(:)
		 dim = ones(1, data.n)
--		 dim(i) = data.sep_parl(i)
--		 K{i}=reshape(p(:), dim)
	 end
 
	return K 
end 

--- (Documentation edited from original, see Mathworks link.)
--
-- This function will separate (do decomposition of) any 2D, 3D or N-D kernel into 1D
-- kernels. Of course, only a sub-set of kernels are separable e.g. a Gaussian kernel,
-- but it will give least-squares solutions for non-separable kernels.
--
-- [K1 KN ERR]=SeparateKernel(H);
--   
-- inputs,
--   H : The 2D, 3D ..., ND kernel
--   
-- outputs,
--   K1 : Cell array with the 1D kernels
--   KN : Approximation of the ND input kernel by the 1D kernels
--   ERR : The sum of absolute difference between approximation and input kernel
--
-- 
-- How the algorithm works:
-- If we have a separable kernel like
-- 
--  H = [1 2 1
--       2 4 2
--       3 6 3];
--
-- We like to solve unknown 1D kernels,
--  a=[a(1) a(2) a(3)]
--  b=[b(1) b(2) b(3)]
--
-- We know that,
--  H = a'*b
--
--      b(1)    b(2)    b(3)
--       --------------------
--  a(1)|h(1,1) h(1,2) h(1,3)
--  a(2)|h(2,1) h(2,2) h(2,3)
--  a(3)|h(3,1) h(3,2) h(3,3)
--
-- Thus,
--  h(1,1) == a(1)*b(1)
--  h(2,1) == a(2)*b(1)
--  h(3,1) == a(3)*b(1)
--  h(4,1) == a(1)*b(2)
-- ...
--
-- We want to solve this by using fast matrix (least squares) math,
--
--  c = M * d; 
--  
--  c a column vector with all kernel values H
--  d a column vector with the unknown 1D kernels 
--
-- But matrices "add" values and we have something like  h(1,1) == a(1)*b(1);
-- We solve this by taking the log at both sides 
-- (We replace zeros by a small value. Whole lines/planes of zeros are
--  removed at forehand and re-added afterwards)
--
--  log( h(1,1) ) == log(a(1)) + log b(1))
--
-- The matrix is something like this,
--
--      a1 a2 a3 b1 b2 b3    
-- M = [1  0  0  1  0  0;  h11
--      0  1  0  1  0  0;  h21
--      0  0  1  1  0  0;  h31
--      1  0  0  0  1  0;  h21
--      0  1  0  0  1  0;  h22
--      0  0  1  0  1  0;  h23
--      1  0  0  0  0  1;  h31
--      0  1  0  0  0  1;  h32
--      0  0  1  0  0  1]; h33
--
-- Least squares solution
--  d = exp(M\log(c))
--
-- with the 1D kernels
--
--  [a(1);a(2);a(3);b(1);b(2);b(3)] = d
--
-- The Problem of Negative Values!!!
--
-- The log of a negative value is possible it gives a complex value, log(-1) = i*pi
-- if we take the expontential it is back to the old value, exp(i*pi) = -1 
--
--  But if we use the solver with on of the 1D vectors we get something like, this :
--
--  input         result        abs(result)    angle(result) 
--   -1     -0.0026 + 0.0125i     0.0128         1.7744 
--    2      0.0117 + 0.0228i     0.0256         1.0958 
--   -3     -0.0078 + 0.0376i     0.0384         1.7744  
--    4      0.0234 + 0.0455i     0.0512         1.0958
--    5      0.0293 + 0.0569i     0.0640         1.0958
-- 
-- The absolute value is indeed correct (difference in scale is compensated
-- by the order 1D vectors)
--
-- As you can see the angle is correlated with the sign of the values. But I
-- didn't found the correlation yet. For some matrices it is something like
--
--  sign=mod(angle(solution)*scale,pi) == pi/2;
--
-- In the current algorithm, we just flip the 1D kernel values one by one.
-- The sign change which gives the smallest error is permanently swapped. 
-- Until swapping signs no longer decreases the error
--
-- Examples,
--   a=permute(rand(5,1),[1 2 3 4])-0.5;
--   b=permute(rand(5,1),[2 1 3 4])-0.5;
--   c=permute(rand(5,1),[3 2 1 4])-0.5;
--   d=permute(rand(5,1),[4 2 3 1])-0.5;
--   H = repmat(a,[1 5 5 5]).*repmat(b,[5 1 5 5]).*repmat(c,[5 5 1 5]).*repmat(d,[5 5 5 1]);
--   [K,KN,err]=SeparateKernel(H);
--   disp(['Summed Absolute Error between Real and approximation by 1D filters : ' num2str(err)]);
--
--   a=permute(rand(3,1),[1 2 3])-0.5;
--   b=permute(rand(3,1),[2 1 3])-0.5;
--   c=permute(rand(3,1),[3 2 1])-0.5;
--   H = repmat(a,[1 3 3]).*repmat(b,[3 1 3 ]).*repmat(c,[3 3 1 ])
--   [K,KN,err]=SeparateKernel(H); err
--
--   a=permute(rand(4,1),[1 2 3])-0.5;
--   b=permute(rand(4,1),[2 1 3])-0.5;
--   H = repmat(a,[1 4]).*repmat(b,[4 1]);
--   [K,KN,err]=SeparateKernel(H); err
--
-- Function is written by D.Kroon, uses "log idea" from A. J. Hendrikse, University of Twente
-- (July 2010).
-- @array kernel 2D, 3D ..., N-D kernel.
-- @uint kcols Number of columns in _kernel_. (TODO: Actually do 3D, etc.?)
-- @treturn array Cell array with the 1D kernels. DOCMEMORE!
-- @treturn array Approximation of _kernel_ by the 1D kernels. DOCMEMORE?
-- @treturn number The sum of absolute difference between approximation and _kernel_.
function M.SeparateKernel (kernel, kcols)
	-- We first make some structure which contains information about the transformation from
	-- kernel to 1D kernel array, number of dimensions, and other stuff.
--	data = InitializeDataStruct(kernel)

	-- Make the matrix of c = M * d.
--	M = MakeMatrix(data)

	-- Solve c = M * d with least squares.
--	par = exp(M \ log(abs(data.H(:))))

	-- Improve the values by solving the remaining difference.
--	KN = Filter1DToFilterND(par, data)
--	par2 = exp(M \ log(abs(KN(:) ./ data.H(:))))
--	par = par ./ par2

	-- Change the sign of a 1D filtering value if it decrease the error.
--	par = FilterCorrSign(par, data)

	-- Split the solution d in separate 1D kernels.
--	K1 = ValueListToFilter1D(par, data)

	-- Re-add the removed zero rows/planes to the 1D vectors.
--	K1 = ReaddZeroRows(data, K1);

	-- Calculate the approximation of the ND kernel if using the 1D kernels.
--	KN = Filter1DToFilterND(par, data, K1);

	-- Calculate the absolute error.
--	ERR = sum(abs(H(:) - KN(:)));
end

-- Export the module.
return M