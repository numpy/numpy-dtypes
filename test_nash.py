#!/usr/bin/env python

from __future__ import division,absolute_import

from numpy import *
from nash import *
from util import *
from fractions import Fraction

F = Fraction

def test_simplex():
    c = -fractions([2,3,4])
    A = fractions([[3,2,1],
                   [2,5,3]])
    b = fractions([10,15])
    f,x = simplex_method(c,A,b) 
    assert f==F(-130,7)
    assert all(x==[F(15,7),0,F(25,7)])

def test_nash():
    random.seed(647121)
    m,n = 3,4
    for _ in xrange(10):
        payoff = fractions(random.randint(-10,10,size=(m,n)))
        A,alice,bob = zero_sum_nash_equilibrium(payoff)
        assert A==dot(payoff,bob).max() # Can Alice do any better?
        assert A==dot(payoff.T,alice).min() # Can Bob do any better?

if __name__=='__main__':
    test_simplex()
    test_nash()
