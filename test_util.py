#!/usr/bin/env python

from __future__ import division

from numpy import *
from util import *
from scipy import sparse

def test_lp():
    c = -array([2.,3,4])
    A = array([[3.,2,1],
               [2,5,3]])
    b = array([10.,15])
    G = -sparse.eye(3,3).tocsr()
    h = zeros(3)
    S = cvxopt_lp(c,G,h,A,b)
    x = S['x']
    assert S['status']=='optimal'
    assert allclose(dot(c,x),-130/7)
    assert allclose(x,[15/7,0,25/7])

if __name__=='__main__':
    test_lp()
