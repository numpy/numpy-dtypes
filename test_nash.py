#!/usr/bin/env python

from __future__ import division,absolute_import

from numpy import *
from nash import *
from util import *
from rational import *

R = rational

def test_simplex():
    c = -rationals([2,3,4])
    A = rationals([[3,2,1],
                   [2,5,3]])
    b = rationals([10,15])
    f,x = simplex_method(c,A,b) 
    assert f==R(-130,7)
    assert all(x==[R(15,7),0,R(25,7)])

def test_nash():
    random.seed(647121)
    m,n = 3,4
    for _ in xrange(10):
        payoff = rationals(random.randint(-10,10,size=(m,n)))
        A,alice,bob = zero_sum_nash_equilibrium(payoff)
        assert A==dot(payoff,bob).max() # Can Alice do any better?
        assert A==dot(payoff.T,alice).min() # Can Bob do any better?

if __name__=='__main__':
    test_simplex()
    test_nash()
