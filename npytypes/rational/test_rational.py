#!/usr/bin/env python

from __future__ import division
from numpy import *
from numpy.testing import assert_
from rational import *

R = rational

def test_misc():
    x = R()
    y = R(7)
    z = R(-6,-10)
    assert_(not x)
    assert_(y and z)
    assert_(z.n is 3)
    assert_(z.d is 5)
    assert_(str(y)=='7')
    assert_(str(z)=='3/5')
    assert_(repr(y)=='rational(7)')
    assert_(repr(z)=='rational(3,5)')

def test_parse():
    assert_(rational("4")==4)
    assert_(rational(" -4 ")==-4)
    assert_(rational("3/5")==R(3,5))
    assert_(rational("  -3/5 ")==R(-3,5))
    for s in '-4 5','1/0','1/-1','1/':
        try:
            rational(s)
            assert_(False)
        except ValueError:
            pass

def test_compare():
    random.seed(1262081)
    for _ in xrange(100):
        xn,yn = random.randint(-10,10,2)
        xd,yd = random.randint(1,10,2)
        x,y = R(xn,xd),R(yn,yd)
        assert_(bool(x)==bool(xn))
        assert_((x==y)==(xn*yd==yn*xd))
        assert_((x<y)==(xn*yd<yn*xd))
        assert_((x>y)==(xn*yd>yn*xd))
        assert_((x<=y)==(xn*yd<=yn*xd))
        assert_((x>=y)==(xn*yd>=yn*xd))
        # Not true in general, but should be for this sample size
        assert_((hash(x)==hash(y))==(x==y))

def test_arithmetic():
    random.seed(1262081)
    for _ in xrange(100):
        xn,yn,zn = random.randint(-100,100,3)
        xd,yd,zd = [n if n else 1 for n in random.randint(-100,100,3)]
        x,y,z = R(xn,xd),R(yn,yd),R(zn,zd)
        assert_(-x==R(-xn,xd))
        assert_(+x is x)
        assert_(--x==x)
        assert_(x+y==R(xn*yd+yn*xd,xd*yd))
        assert_(x+y==x--y==R(xn*yd+yn*xd,xd*yd))
        assert_(-x+y==-(x-y))
        assert_((x+y)+z==x+(y+z))
        assert_(x*y==R(xn*yn,xd*yd))
        assert_((x*y)*z==x*(y*z))
        assert_(-(x*y)==(-x)*y)
        assert_(x*y==y*x)
        assert_(x*(y+z)==x*y+x*z)
        if y:
            assert_(x/y==R(xn*yd,xd*yn))
            assert_(x/y*y==x)
            assert_(x//y==xn*yd//(xd*yn))
            assert_(x%y==x-x//y*y)
        assert_(x+7==7+x==x+R(7))
        assert_(x*7==7*x==x*R(7))
        assert_(int(x)==int(xn/xd))
        assert_(allclose(float(x),xn/xd))
        assert_(abs(x)==R(abs(xn),abs(xd)))
        # TODO: test floor, ceil, abs

def test_errors():
    # Check invalid constructions
    for args in (R(3,2),4),(1.2,),(1,2,3):
        try:
            R(*args)
            assert_(False)
        except TypeError:
            pass
    for args in (1<<80,),(2,1<<80):
        try:
            R(*args)
            assert_(False)
        except OverflowError:
            pass
    # Check for zero divisions
    try:
        R(1,0)
        assert_(False)
    except ZeroDivisionError:
        pass
    try:
        R(7)/R()
        assert_(False)
    except ZeroDivisionError:
        pass
    # Check for LONG_MIN overflows
    for args in (-1<<63,-1),(1<<63,):
        try:
            R(*args)
            assert_(False)
        except OverflowError:
            pass
    # Check for overflow in addition
    r = R(1<<30)
    try:
        r+r
        assert_(False)
    except OverflowError:
        pass
    # Check for overflow in multiplication
    # Twin primes from http://primes.utm.edu/lists/small/10ktwins.txt
    p = R(1262081,1262083)
    r = p
    for _ in xrange(int(log(2.**31)/log(r.d))-1):
        r *= p
    try:
        r*p
        assert_(False)
    except OverflowError:
        pass
    # Float/rational arithmetic should fail
    for x,y in (.2,R(3,2)),(R(3,2),.2):
        try:
            x+y
            assert_(False)
        except TypeError:
            pass

def test_numpy_basic():
    d = dtype(rational)
    assert_(d.itemsize==8)
    x = zeros(5,d)
    assert_(type(x[2]) is rational)
    assert_(x[3]==0)
    assert_(ones(5,d)[3]==1)
    x[2] = 2
    assert_(x[2]==2)
    x[3] = R(4,5)
    assert_(5*x[3]==4)
    try:
        x[4] = 1.2
        assert_(False)
    except TypeError:
        pass
    i = arange(R(1,3),R(5,3),R(1,3))
    assert_(i.dtype is d)
    assert_(all(i==[R(1,3),R(2,3),R(3,3),R(4,3)]))
    assert_(numerator(i).dtype==denominator(i).dtype==dtype(int64))
    assert_(all(numerator(i)==[1,2,1,4]))
    assert_(all(denominator(i)==[3,3,1,3]))
    y = zeros(4,d)
    y[1:3] = i[1:3] # Test unstride copyswapn
    assert_(all(y==[0,R(2,3),R(3,3),0]))
    assert_(all(nonzero(y)[0]==(1,2)))
    y[::3] = i[:2] # Test strided copyswapn
    assert_(all(y==[R(1,3),R(2,3),R(3,3),R(2,3)]))
    assert_(searchsorted(arange(0,20),R(7,2))==4) # Test compare
    assert_(argmin(y)==0)
    assert_(argmax(y)==2)
    assert_(y.min()==R(1,3))
    assert_(y.max()==1)
    assert_(dot(i,y)==R(22,9))
    y[:] = 7 # Test fillwithscalar
    assert_(all(y==7))

def test_numpy_cast():
    r = arange(R(10,3),step=R(1,3),dtype=rational)
    # Check integer to rational conversion
    for T in int8,int32,int64:
        n = arange(10,dtype=T)
        assert_(all(n.astype(rational)==3*r))
        assert_(all(n+r==4*r))
    # Check rational to integer conversion
    assert_(all(r.astype(int)==r.astype(float).astype(int)))
    # Check detection of overflow during casts
    for x in array(1<<40),array([1<<40]):
        try:
            x.astype(int64).astype(rational)
            assert_(False)
        except OverflowError:
            pass
    # Check conversion to and from floating point
    for T in float,double:
        f = arange(10,dtype=float)/3
        assert_(allclose(r.astype(float),f))
        rf = r+f
        assert_(rf.dtype==dtype(float))
        assert_(allclose(rf,2*f))
        try:
            f.astype(rational)
            assert_(False)
        except ValueError:
            pass

def test_numpy_ufunc():
    d = dtype(rational)
    # Exhaustively check arithmetic for all small numerators and denominators
    N = arange(-10,10).reshape(-1,1)
    D = arange(1,10).reshape(1,-1)
    x = (N.astype(rational)/D).reshape(-1,1)
    y = x.reshape(1,-1)
    s = y+(y==0)
    xf = x.astype(float)
    for f in add,subtract,multiply,minimum,maximum,divide,true_divide:
        z = s if f in (divide,true_divide) else y
        fxy = f(x,z)
        assert_(fxy.dtype is d)
        assert_(allclose(fxy,f(xf,z)))
    assert_(all(x//s==floor(x/s)))
    assert_(all(x%s==x-x//s*s))
    xn,yn = numerator(x),numerator(y)
    xd,yd = denominator(x),denominator(y)
    for f in equal,not_equal,less,greater,less_equal,greater_equal:
        assert_(all(f(x,y)==f(xn*yd,yn*xd)))
    for f in negative,absolute,floor,ceil,trunc,square,sign:
        fx = f(x)
        assert_(fx.dtype is d)
        assert_(allclose(fx,f(xf)))
    assert_(all(denominator(rint(x))==1))
    assert_(all(absolute(rint(x)-x)<=R(1,2)))
    assert_(all(reciprocal(s)*s==1))
    # Check that missing ufuncs promote to float
    r = array([R(5,3)])
    assert_(r.dtype==dtype(rational))
    assert_(sin(r).dtype==dtype(float))

def test_gcd_lcm():
    x = arange(-10,10).reshape(-1,1)
    y = x.reshape(1,-1)
    z = x.reshape(-1,1,1)
    g = gcd(x,y)
    l = lcm(x,y)
    assert_(all(g*l==absolute(x*y)))
    assert_(all(gcd(x,lcm(y,z))==lcm(gcd(x,y),gcd(x,z))))
    assert_(all(gcd(2,[1,2,3,4,5,6])==[1,2,1,2,1,2]))
    assert_(all(lcm(2,[1,2,3,4,5,6])==[2,2,6,4,10,6]))
    assert_(lcm.reduce(arange(1,10))==2520)

def test_numpy_errors():
    # Check that exceptions inside ufuncs are detected
    r = array([1<<30]).astype(rational)
    try:
        r+r
        assert_(False)
    except OverflowError:
        pass
    r = zeros(3,rational)
    try:
        reciprocal(r)
        assert_(False)
    except ZeroDivisionError:
        pass

if __name__=='__main__':
    test_parse()
    test_numpy_cast()
    test_numpy_basic()
