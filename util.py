'''Miscellaneous utilities'''

from __future__ import division,absolute_import

from numpy import *
from rational import *
from scipy import sparse
import tempfile
import subprocess

def amap(f,x):
    x = asanyarray(x)
    return array(map(f,x.ravel())).view(type(x)).reshape(x.shape)

def rationals(x):
    return asarray(x).astype(rational)

def sparse_save(file,**kwargs):
    d = {}
    for k,v in kwargs.items():
        if isinstance(v,(ndarray,str,generic)):
            d[k] = v
        elif isinstance(v,sparse.spmatrix):
            v = v.tocsr()
            d[k+'/data'] = v.data 
            d[k+'/indices'] = v.indices
            d[k+'/offsets'] = v.indptr
            d[k+'/shape'] = v.shape
        else:
            raise TypeError('unknown type %s'%type(v).__name__)
    savez(file,**d)

def sparse_load(file):
    data = load(file)
    d = {}
    for k,v in data.items():
        if k.endswith('/data'):
            k = k[:-5]
            d[k] = sparse.csr_matrix((v,data[k+'/indices'],data[k+'/offsets']),data[k+'/shape'])
        elif k.endswith('/indices') or k.endswith('/offsets') or k.endswith('/shape'):
            pass
        else:
            d[k] = v
    return d

def cvxopt_lp(c,G,h,A=None,b=None):
    assert (A is None)==(b is None)
    if A is None:
        A = zeros((0,len(c)))
        b = zeros(0)
    assert G.shape==(len(h),len(c))
    assert A.shape==(len(b),len(c))
    input = tempfile.NamedTemporaryFile(prefix='cvxopt-in',suffix='.npz') 
    output = tempfile.NamedTemporaryFile(prefix='cvxopt-out',suffix='.npz') 
    sparse_save(input.name,c=c,G=G,h=h,A=A,b=b)
    cmd = ['./cvxopt',input.name,output.name]
    r = subprocess.call(cmd)
    if r:
        raise RuntimeError('cmd failed: status %d'%r)
    data = sparse_load(output.name)
    if 'error' in data:
        raise RuntimeError('lp solve failed: %s'%data['error'])
    return data

speye = sparse.eye

def spdiag(x):
    data = []
    indices = []
    offsets = []
    shape = array([0,0],dtype=int32)
    total = array([0],dtype=int32)
    for x in x:
        x = sparse.csr_matrix(x)
        data.append(x.data)
        indices.append(x.indices+shape[1])
        offsets.append(x.indptr[:-1]+total[0])
        shape += x.shape
        total += x.indptr[-1]
    return sparse.csr_matrix((hstack(data),hstack(indices),hstack(offsets+[total])),shape=shape)

def spzeros(m,n):
        return sparse.csr_matrix((m,n))

def speye(m,n=None):
    if n is None:
        n = m
    return sparse.eye(m,n)

def asplit(x,*sizes):
    if sum(sizes)!=len(x):
        raise IndexError('expected size %s = %d, got %d'%('+'.join(map(str,sizes)),sum(sizes),len(x)))
    r = []
    n = 0
    for s in sizes:
        r.append(x[n:n+s])
        n += s
    assert n==len(x)
    return r
