'''Exact Nash equilibrium computation given a payoff matrix'''

from __future__ import division
from numpy import *

'''We implement a fairly unoptimized version of the simplex method for linear programming following
Wikipedia: http://en.wikipedia.org/wiki/Simplex_method.  Thus, our tableu has the structure
    T = [[1,-cB.T,-cD.T,zB]
        ,[0, 1,     D,   b]]
where B and N are the ordered lists of basis and nonbasis variables, respectively.  Note that we
store the identity matrix explicitly, which is lazy.

One note: since we always operate on linear programs in canonical form, where the only inequality
constraints are x>=0, the feasible solutions we traverse always have m-n zeros and n (potential)
nonzeros, where the constraint matrix A in A x = b is m by n.  Thus, the core state of the simplex
method is the set of n possibly nonzero variable B, called the set of basis variables.
'''

class Unbounded(OverflowError):
    pass

class Infeasible(ArithmeticError):
    pass

def solve_tableau(T,B,N):
    # Verify that T has an identity matrix where we expect it to be
    k = len(T)-len(B) # 1 for phase 2, 2 for phase 1
    assert all(T[k:,k+B]==eye(len(B),dtype=T.dtype))
    # Price out the basis variables
    T[0] -= dot(T[k:].T,T[0,k+B])
    assert all(T[0,k+B]==0)
    # Solve the tableu
    while 1:
        # Assert that we're sitting on a feasible basic solution
        assert all(T[k:,-1]>=0)
        # Pick a variable to enter the basis using Dantzig's rule
        enter = argmax(T[0,k+N])
        c = k+N[enter]
        if T[0,c]<=0:
            break # We've hit a local optimum, which is therefore a global optimum
        # Pick a variable to leave
        leavings, = nonzero(T[k:,c]>0)
        if not len(leavings):
            raise Unbounded('unbounded linear program')
        rows = k+leavings
        leave = leavings[argmin(T[rows,-1]/T[rows,c])]
        r = k+leave
        # Perform the pivot
        T[r] /= T[r,c]
        rows, = nonzero(arange(len(T))!=r)
        T[rows] -= T[rows,c].reshape(-1,1).copy()*T[r]
        # Update the sets of basis and nonbasis variables
        N[enter],B[leave] = B[leave],N[enter]

def simplex_method(c,A,b):
    '''Minimize dot(c,x) s.t. Ax = b, x >= 0 using the simplex method, and return dot(c,x),x.
    If c,A,b are fractions, the result is exact.'''
    # Phase 1: Add slack variables to get an initial canonical tableu, and solve
    (n,m),dtype = A.shape,A.dtype
    assert c.shape==(m,)
    assert b.shape==(n,)
    T = vstack([hstack([1,zeros(m+1,dtype),-ones(n,dtype),0]).reshape(1,-1),
                hstack([0,1,-c,zeros(n+1,dtype)]).reshape(1,-1),
                hstack([zeros((n,2),dtype),A,eye(n,dtype=dtype),b.reshape(-1,1)])])
    N = arange(m)
    B = m+arange(n)
    solve_tableau(T,B,N)
    # Check for infeasibility
    if T[0,-1]<0:
        raise Infeasible('infeasible linear program')
    # Verify that the auxiliary slack variables are nonbasic.  This is not always the
    # case--they could be basic but just happen to be zero--but we'll deal with that later.
    assert B.max()<m
    # Remove the now zero auxiliary variables
    T = T[1:,hstack([1+arange(m+1),-1])]
    assert T.shape==(1+n,1+m+1)
    N = N[nonzero(N<m)[0]]
    # Solve our new canonical tableau
    solve_tableau(T,B,N)
    x = zeros(m,dtype)
    x[B] = T[1:,-1]
    return T[0,-1],x

def zero_sum_nash_equilibrium_side(payoff):
    '''Alice chooses the row and maximizes, Bob chooses the column and minimizes.
    Given Alice's payoff matrix, we compute Alice's optimal payoff and (mixed) strategy.
    See http://en.wikipedia.org/wiki/Zero_sum_game for details.'''
    assert payoff.ndim==2
    M = payoff.T
    M = M - M.min()
    M = M/(M.max() or 1)+1
    assert all(M>0)
    # We want to minimize sum(u) s.t. Mu >= 1, u >= 0.  Let M be m by n.
    # Adding m positive slack variables s, this is
    #     min 1_n . u s.t. Mu = 1_m + s, [u,s] >= 0
    #     min 1_n . u s.t. Mu - s = 1_m, [u,s] >= 0
    #     min hstack(1_n,0_m) . [u,s] s.t. hstack(M,-eye(m)) stack(u,s) = 1_m, [u,s] >= 0
    # Our linear program is now in standard form.
    m,n = M.shape
    dtype = M.dtype
    f,u = simplex_method(hstack([ones(n,dtype),zeros(m,dtype)]),hstack([M,-eye(m,dtype=dtype)]),ones(m,dtype))
    u = u[:n]
    u /= sum(u)
    return u

def zero_sum_nash_equilibrium(payoff):
    alice = zero_sum_nash_equilibrium_side(payoff)
    bob = zero_sum_nash_equilibrium_side(-payoff.T)
    return dot(payoff,bob).max(),alice,bob
