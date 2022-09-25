# PyPhot

The Python Imaging and Photometry Reduction Pipeline. 

# Instruments Served
* CFHT/WIRCam
* LBT/LBC
* Magellan/IMACS
* MMT/MMIRS
* Keck/LRIS
* Keck/NIRES Acquisition Camera

# Data Processing Steps
* Build master calibrations
  
 - Bias, Dark, Pixel Flat, Illuminating Flat

* Image processing
  
 - Removing detector effects, including gain correction, bias subtraction, dark subtraction, and flat fielding

 - Supersky flattening, fringe subtraction, and background subtraction
   
* Post processing

 - Astrometric calibration

 - Calibrating zeropoint for individual detector chips

 - Coadding images target by target

 - Extracting photometric catalogs 

# Software Requirements
#### We strongly encourage you to install the following with conda
* Python
* SExtractor
* SWarp
* Scamp

# Astromatic software installation
* conda install -c conda-forge astromatic-source-extractor
* conda install -c conda-forge astromatic-scamp
* conda install -c conda-forge astromatic-swarp

# Python package requirements
(see `pyphot/requirements.txt`)
* astropy
* astroquery
* configobj
* matplotlib
* numpy
* photutils
* scipy

Note that this package uses many bookkeeping stuff from PyPeIt, 
so it will look familiar to those PyPeIt funs when running the scripts. 
Since it is not easy to import those bookkeeping related functions directly from PyPeIt, 
we copied and modified them into this package. The copyright of these functions 
belongs to the PyPeIt team. 

https://github.com/pypeit/PypeIt

# License (BSD-3)

Copyright (c) 2021-2022, PyPhot Developers All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

 - Neither the name of the Team nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


