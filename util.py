'''Miscellaneous utilities'''

from __future__ import division,absolute_import

from numpy import *
from fractions import Fraction

def amap(f,x):
    x = asanyarray(x)
    return array(map(f,x.ravel())).view(type(x)).reshape(x.shape)

class fractions(ndarray):
    def __new__(cls,x):
        return amap(Fraction,asarray(x,object).view(cls))

    #def __str__(self):
    #    n = max(len(str(f)) for f in self.ravel())
    #    def pad(f):
    #        return '%*s'%(n,f)
    #    return ndarray.__str__(amap(pad,self))
