signal_ops
==========

Submodule implementing various operations on discrete signals, e.g. convolution.

**STATUS**
==========

Basic convolution is mostly functional, though the comments and docs could stand some improvement.

Correlation support is planned, though the details are not yet final. I don't know if it warrants non-FFT modules as
I did with convolution. At the very least, it seems something obvious to add support for it in precomputed kernels.

The overlap-* algorithms are probably worthy of their own module. I still mean to add the 2D variants, plus perhaps
a pseudo-2D version (where one dimension is large, the other not so much, thus not requiring overlap in both).

Maybe it would be worth providing built-in filters. However, this might not be the place for them.

**DEPENDENCIES**
================

Tested on Lua 5.1. (Some tests [here](https://github.com/ggcrunchy/Strays/blob/master/Unit%20Tests/convolve.lua).)

The `fft_convolution` module depends on the [DFT submodule](https://github.com/ggcrunchy/dft_ops).