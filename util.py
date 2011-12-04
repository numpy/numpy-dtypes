'''Miscellaneous utilities'''

from __future__ import division,absolute_import

from numpy import *
from rational import *

def amap(f,x):
    x = asanyarray(x)
    return array(map(f,x.ravel())).view(type(x)).reshape(x.shape)

def rationals(x):
    return asarray(x).astype(rational)
