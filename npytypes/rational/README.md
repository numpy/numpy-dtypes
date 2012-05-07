Test code for user defined types in Numpy
=========================================

This code implements fixed precision rational numbers exposed to Python as
a test of user-defined type support in Numpy.

### Dependencies

* [python](http://python.org)
* [numpy](http://numpy.scipy.org): Requires a patched version (see below)
* [py.test](http://pytest.org): To run tests

On a Mac, these can be obtained through [MacPorts](http://www.macports.org) via

    sudo port install py26-numpy py26-py

The original version of this code exposed several bugs in the (hopefully) old
version of numpy, so if it fails try upgrading numpy.  My branch which fixed
all but one of the bugs uncovered is here:

    https://github.com/girving/numpy/tree/fixuserloops

These changes should be incorporated in the main numpy git repo fairly soon.

### Usage

To build and run the tests, do

    make
    py.test
