========
speedynn
========

The speedynn package is a Python library that aims at accelerating feature selection for nearest neighbor models via modern many-core devices such as graphics processing units (GPUs). The implementation is based on [OpenCL](https://www.khronos.org/opencl/OpenCL). 

============
Dependencies
============

The speedynn package is tested under Python 2.6 and Python 2.7. The required Python dependencies are:

- NumPy >= 1.6.1

Further, [Swig](http://www.swig.org), [OpenCL](https://www.khronos.org/opencl/OpenCL), [setuptools](https://pypi.python.org/pypi/setuptools), and a working C/C++ compiler need to be available.

==========
Quickstart
==========

The package can be installed by executing the Makefile in the speedynn subdirectory, i.e.:
```
cd speedynn
make
```
Further, the package root needs to be added to the PYTHONPATH. On Debian/Ubuntu systems, this line should be added to .bashrc:
```
export PYTHONPATH=$PYTHONPATH:~/speedynn
```
in case the speedynn directory is located in the home folder. We recommend to use [virtualenv](https://pypi.python.org/pypi/virtualenv) to generate an independent environment for the package.

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv2). The authors are not responsible for any implications that stem from the use of this software.

