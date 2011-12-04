// Fixed size rational numbers exposed to Python

#define NPY_NO_DEPRECATED_API

#include <stdint.h>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <typeinfo>
#include <iostream>
#include <Python/Python.h>
#include <Python/structmember.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

namespace {

using std::abs;
using std::min;
using std::max;
using std::cout;
using std::endl;
using std::swap;
using std::type_info;
using std::exception;
using std::numeric_limits;

template<bool c> struct assertion_helper {};
template<> struct assertion_helper<true> { typedef int type; };
#define static_assert(condition) \
    typedef assertion_helper<(condition)!=0>::type assertion_typedef_##__LINE__

typedef __int128_t int128_t;
typedef __uint128_t uint128_t;

// For now, we require a 64-bit machine
static_assert(sizeof(void*)==8);

// Relevant arithmetic exceptions

class overflow : public exception {
};

class zero_divide : public exception {
};

// Integer arithmetic utilities

template<class I> inline I safe_neg(I x) {
    if (x==numeric_limits<I>::min())
        throw overflow();
    return -x;
}

template<class I> inline I safe_abs(I x) {
    if (x>=0)
        return x;
    I nx = -x;
    if (nx<0)
        throw overflow();
    return nx;
}

// Check for negative numbers without compiler warnings for unsigned types
#define DEFINE_IS_NEGATIVE(bits) \
    inline bool is_negative(int##bits##_t x) { return x<0; } \
    inline bool is_negative(uint##bits##_t x) { return false; }
DEFINE_IS_NEGATIVE(8)
DEFINE_IS_NEGATIVE(16)
DEFINE_IS_NEGATIVE(32)
DEFINE_IS_NEGATIVE(64)
DEFINE_IS_NEGATIVE(128)
inline bool is_negative(long x) { return x<0; }

template<class dst,class src> inline dst safe_cast(src x) {
    dst y = x;
    if (x != (src)y || is_negative(x)!=is_negative(y))
        throw overflow();
    return y;
}

template<class I> I gcd(I x, I y) {
    x = safe_abs(x);
    y = safe_abs(y);
    if (x < y)
        swap(x,y);
    while (y) {
        x = x%y;
        swap(x,y);
    }
    return x;
}

template<class I> I lcm(I x, I y) {
    if (!x || !y)
        return 0;
    x /= gcd(x,y);
    I lcm = x*y;
    if (lcm/y!=x)
        throw overflow();
    return safe_abs(lcm);
}

// Fixed precision rational numbers

class rational {
public:
    typedef int64_t I; // The type of numerators and denominators
    typedef int128_t DI; // Double-wide integer type for detecting overflow

    I n; // numerator
    I dmm; // denominator minus one: numpy.zeros() uses memset(0) for non-object types, so need to ensure that rational(0) has all zero bytes

    rational(I n=0)
        :n(n),dmm(0) {}

    template<class SI> rational(SI n_, SI d_) {
        if (!d_)
            throw zero_divide();
        SI g = gcd(n_,d_);
        n = safe_cast<I>(n_/g);
        I d = safe_cast<I>(d_/g);
        if (d <= 0) {
            d = -d;
            n = safe_neg(n);
        }
        dmm = d-1;
    }

    I d() const {
        return dmm+1;
    }

private:
    struct fast {};

    // Private to enforce that d > 0
    template<class SI> rational(SI n_, SI d_, fast) {
        SI g = gcd(n_,d_);
        n = safe_cast<I>(n_/g);
        dmm = safe_cast<I>(d_/g)-1;
    }
public:

    rational operator-() const {
        rational x;
        x.n = safe_neg(n);
        x.dmm = dmm;
        return x;
    }

    rational operator+(rational x) const {
        // Note that the numerator computation can never overflow int128_t, since each term is strictly under 2**128/4 (since d > 0).
        return rational(DI(n)*x.d()+DI(d())*x.n,DI(d())*x.d(),fast());
    }

    rational operator-(rational x) const {
        // We're safe from overflow as with +
        return rational(DI(n)*x.d()-DI(d())*x.n,DI(d())*x.d(),fast());
    }

    rational operator*(rational x) const {
        return rational(DI(n)*x.n,DI(d())*x.d(),fast());
    }

    rational operator/(rational x) const {
        return rational(DI(n)*x.d(),DI(d())*x.n);
    }

    friend I floor(rational x) {
        // Always round down
        if (x.n>=0)
            return x.n/x.d();
        // This can be done without casting up to 128 bits, but it requires working out all the sign cases
        return -((-DI(x.n)+x.d()-1)/x.d());
    }

    friend I ceil(rational x) {
        return -floor(-x);
    }

    rational operator%(rational x) const {
        return *this-x*floor(*this/x);
    }

    friend rational abs(rational x) {
        rational y;
        y.n = safe_abs(x.n);
        y.dmm = x.dmm;
        return y;
    }

    friend I rint(rational x) {
        // Round towards nearest integer, moving exact half integers towards zero
        I d = x.d();
        return (2*DI(x.n)+(x.n<0?-d:d))/(2*DI(d)); 
    }

    friend int sign(rational x) {
        return x.n<0?-1:x.n==0?0:1;
    }

    friend rational inverse(rational x) {
        if (!x.n)
            throw zero_divide();
        rational y;
        y.n = x.d();
        I d = x.n;
        if (d <= 0) {
            d = safe_neg(d);
            y.n = -y.n;
        }
        y.dmm = d-1;
        return y;
    }

private:
    struct unusable { void f(){} };
    typedef void (unusable::*safe_bool)();
public:

    operator safe_bool() const { // allow conversion to bool without allowing conversion to T
        return n?&unusable::f:0;
    }
};

inline bool operator==(rational x, rational y) {
    // Since we enforce d > 0, and store fractions in reduced form, equality is easy.
    return x.n==y.n && x.dmm==y.dmm;
}

inline bool operator!=(rational x, rational y) {
    return !(x==y);
}

inline bool operator<(rational x, rational y) {
    typedef rational::DI DI;
    return DI(x.n)*y.d() < DI(y.n)*x.d();
}

inline bool operator>(rational x, rational y) {
    return y<x;
}

inline bool operator<=(rational x, rational y) {
    return !(y<x);
}

inline bool operator>=(rational x, rational y) {
    return !(x<y);
}

template<class D,class S> inline D cast(S x);

#define DEFINE_INT_CAST(I) \
    template<> inline I cast(rational x) { return safe_cast<I>(x.n/x.d()); } \
    template<> inline rational cast(I n) { return safe_cast<int64_t>(n); }
DEFINE_INT_CAST(int8_t)
DEFINE_INT_CAST(uint8_t)
DEFINE_INT_CAST(int16_t)
DEFINE_INT_CAST(uint16_t)
DEFINE_INT_CAST(int32_t)
DEFINE_INT_CAST(uint32_t)
DEFINE_INT_CAST(int64_t)
DEFINE_INT_CAST(uint64_t)
template<> inline long cast(rational x) { return safe_cast<long>(x.n/x.d()); }

template<> inline float  cast(rational x) { return (float) x.n/x.d(); }
template<> inline double cast(rational x) { return (double)x.n/x.d(); }

template<> inline bool cast(rational x) { return x.n!=0; }
template<> inline rational cast(bool b) { return b; }

bool scan_rational(const char*& s, rational& x) {
    long n,d;
    int offset;
    if (!sscanf(s,"%ld%n",&n,&offset))
        return false;
    const char* ss = s+offset;
    if (*ss!='/') {
        s = ss;
        x = n;
        return true;
    }
    ss++;
    if (!sscanf(ss,"%ld%n",&d,&offset) || d<=0)
        return false;
    s = ss+offset;
    x = rational(n,d);
    return true;
}

// Conversion from C++ to Python exceptions

void set_python_error(const exception& e) {
    // Numpy will call us without access to the Python API, so we need to grab it before setting an error
    PyGILState_STATE state = PyGILState_Ensure();
    const type_info& type = typeid(e);
    if (type==typeid(overflow))
        PyErr_SetString(PyExc_OverflowError,"overflow in rational arithmetic");
    else if (type==typeid(zero_divide))
        PyErr_SetString(PyExc_ZeroDivisionError,"zero divide in rational arithmetic");
    else
        PyErr_Format(PyExc_RuntimeError,"unknown exception %s: %s",type.name(),e.what());
    PyGILState_Release(state);
}

// Expose rational to Python as a numpy scalar

typedef struct {
    PyObject_HEAD;
    rational r;
} PyRational;

extern PyTypeObject PyRational_Type;

inline bool PyRational_Check(PyObject* object) {
    return PyObject_IsInstance(object,(PyObject*)&PyRational_Type);
}

PyObject* PyRational_FromRational(rational x) {
    PyRational* p = (PyRational*)PyRational_Type.tp_alloc(&PyRational_Type,0);
    if (p)
        p->r = x;
    return (PyObject*)p;
}

PyObject* rational_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError,"constructor takes no keyword arguments");
        return 0;
    }
    Py_ssize_t size = PyTuple_GET_SIZE(args);
    if (size>2) {
        PyErr_SetString(PyExc_TypeError,"expected rational or numerator and optional denominator");
        return 0;
    }
    PyObject* x[2] = {PyTuple_GET_ITEM(args,0),PyTuple_GET_ITEM(args,1)};
    if (size==1) {
        if (PyRational_Check(x[0])) {
            Py_INCREF(x[0]);
            return x[0];
        } else if (PyString_Check(x[0])) {
            const char* s = PyString_AS_STRING(x[0]);
            rational x; 
            if (scan_rational(s,x)) {
                for (const char* p = s; *p; p++)
                    if (!isspace(*p))
                        goto bad;
                return PyRational_FromRational(x);
            }
            bad:
            PyErr_Format(PyExc_ValueError,"invalid rational literal '%s'",s);
            return 0;
        }
    }
    long n[2]={0,1};
    for (int i=0;i<size;i++) {
        n[i] = PyInt_AsLong(x[i]);
        if (n[i]==-1 && PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_TypeError))
                PyErr_Format(PyExc_TypeError,"expected integer %s, got %s",(i?"denominator":"numerator"),x[i]->ob_type->tp_name);
            return 0;
        }
        // Check that we had an exact integer
        PyObject* y = PyInt_FromLong(n[i]);
        if (!y)
            return 0;
        int eq = PyObject_RichCompareBool(x[i],y,Py_EQ);
        Py_DECREF(y);
        if (eq<0)
            return 0;
        if (!eq) {
            PyErr_Format(PyExc_TypeError,"expected integer %s, got %s",(i?"denominator":"numerator"),x[i]->ob_type->tp_name);
            return 0;
        }
    }
    rational r;
    try {
        r = rational(n[0],n[1]);
    } catch (const exception& e) {
        set_python_error(e);
        return 0;
    }
    return PyRational_FromRational(r);
}

// Returns Py_NotImplemented on most conversion failures, or raises an overflow error for too long ints
#define AS_RATIONAL(object) ({ \
    rational r; \
    if (PyRational_Check(object)) \
        r = ((PyRational*)object)->r; \
    else { \
        long n = PyInt_AsLong(object); \
        if (n==-1 && PyErr_Occurred()) { \
            if (PyErr_ExceptionMatches(PyExc_TypeError)) { \
                PyErr_Clear(); \
                Py_INCREF(Py_NotImplemented); \
                return Py_NotImplemented; \
            } \
            return 0; \
        } \
        PyObject* y = PyInt_FromLong(n); \
        if (!y) \
            return 0; \
        int eq = PyObject_RichCompareBool(object,y,Py_EQ); \
        Py_DECREF(y); \
        if (eq<0) \
            return 0; \
        if (!eq) { \
            Py_INCREF(Py_NotImplemented); \
            return Py_NotImplemented; \
        } \
        r = n; \
    } \
    r; })

PyObject* rational_richcompare(PyObject* a, PyObject* b, int op) {
    rational x = AS_RATIONAL(a),
             y = AS_RATIONAL(b);
    bool result = false;
    #define OP(py,op) case py: result = x op y; break;
    switch (op) {
        OP(Py_LT,<)
        OP(Py_LE,<=)
        OP(Py_EQ,==)
        OP(Py_NE,!=)
        OP(Py_GT,>)
        OP(Py_GE,>=)
    };
    #undef OP
    return PyBool_FromLong(result);
}

PyObject* rational_repr(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (x.d()!=1)
        return PyString_FromFormat("rational(%ld,%ld)",(long)x.n,(long)x.d());
    else
        return PyString_FromFormat("rational(%ld)",(long)x.n);
}

PyObject* rational_str(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (x.d()!=1)
        return PyString_FromFormat("%ld/%ld",(long)x.n,(long)x.d());
    else
        return PyString_FromFormat("%ld",(long)x.n);
}

long rational_hash(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    long h = 131071*x.n+524287*x.dmm; // Use a fairly weak hash as Python expects
    return h==-1?2:h; // Never return the special error value -1
}

#define RATIONAL_BINOP_2(name,exp) \
    PyObject* rational_##name(PyObject* a, PyObject* b) { \
        rational x = AS_RATIONAL(a), \
                 y = AS_RATIONAL(b); \
        rational z; \
        try { \
            z = exp; \
        } catch (const exception& e) { \
            set_python_error(e); \
            return 0; \
        } \
        return PyRational_FromRational(z); \
    }
#define RATIONAL_BINOP(name,op) RATIONAL_BINOP_2(name,x op y)
RATIONAL_BINOP(add,+)
RATIONAL_BINOP(subtract,-)
RATIONAL_BINOP(multiply,*)
RATIONAL_BINOP(divide,/)
RATIONAL_BINOP(remainder,%)
RATIONAL_BINOP_2(floor_divide,floor(x/y))

#define RATIONAL_UNOP(name,type,exp,convert) \
    PyObject* rational_##name(PyObject* self) { \
        rational x = ((PyRational*)self)->r; \
        type y; \
        try { \
            y = exp; \
        } catch (const exception& e) { \
            set_python_error(e); \
            return 0; \
        } \
        return convert(y); \
    }
RATIONAL_UNOP(negative,rational,-x,PyRational_FromRational)
RATIONAL_UNOP(absolute,rational,abs(x),PyRational_FromRational)
RATIONAL_UNOP(int,long,cast<long>(x),PyInt_FromLong)
RATIONAL_UNOP(float,double,cast<double>(x),PyFloat_FromDouble)

PyObject* rational_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

int rational_nonzero(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    return x.n!=0;
}

PyNumberMethods rational_as_number = {
    rational_add,          // nb_add
    rational_subtract,     // nb_subtract
    rational_multiply,     // nb_multiply
    rational_divide,       // nb_divide
    rational_remainder,    // nb_remainder
    0,                     // nb_divmod
    0,                     // nb_power
    rational_negative,     // nb_negative
    rational_positive,     // nb_positive
    rational_absolute,     // nb_absolute
    rational_nonzero,      // nb_nonzero
    0,                     // nb_invert
    0,                     // nb_lshift
    0,                     // nb_rshift
    0,                     // nb_and
    0,                     // nb_xor
    0,                     // nb_or
    0,                     // nb_coerce
    rational_int,          // nb_int
    rational_int,          // nb_long
    rational_float,        // nb_float
    0,                     // nb_oct
    0,                     // nb_hex

    0,                     // nb_inplace_add
    0,                     // nb_inplace_subtract
    0,                     // nb_inplace_multiply
    0,                     // nb_inplace_divide
    0,                     // nb_inplace_remainder
    0,                     // nb_inplace_power
    0,                     // nb_inplace_lshift
    0,                     // nb_inplace_rshift
    0,                     // nb_inplace_and
    0,                     // nb_inplace_xor
    0,                     // nb_inplace_or

    rational_floor_divide, // nb_floor_divide
    rational_divide,       // nb_true_divide
    0,                     // nb_inplace_floor_divide
    0,                     // nb_inplace_true_divide
    0,                     // nb_index
};

PyObject* rational_n(PyObject* self, void* closure) {
    return PyInt_FromLong(((PyRational*)self)->r.n);
}

PyObject* rational_d(PyObject* self, void* closure) {
    return PyInt_FromLong(((PyRational*)self)->r.d());
}

PyGetSetDef rational_getset[] = {
    {(char*)"n",rational_n,0,(char*)"numerator",0},
    {(char*)"d",rational_d,0,(char*)"denominator",0},
    {0} // sentinel
};

PyTypeObject PyRational_Type = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,                                        // ob_size
    "rational",                               // tp_name
    sizeof(PyRational),                       // tp_basicsize
    0,                                        // tp_itemsize
    0,                                        // tp_dealloc
    0,                                        // tp_print
    0,                                        // tp_getattr
    0,                                        // tp_setattr
    0,                                        // tp_compare
    rational_repr,                            // tp_repr
    &rational_as_number,                      // tp_as_number
    0,                                        // tp_as_sequence
    0,                                        // tp_as_mapping
    rational_hash,                            // tp_hash
    0,                                        // tp_call
    rational_str,                             // tp_str
    0,                                        // tp_getattro
    0,                                        // tp_setattro
    0,                                        // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, // tp_flags
    "Fixed precision rational numbers",       // tp_doc
    0,                                        // tp_traverse
    0,                                        // tp_clear
    rational_richcompare,                     // tp_richcompare
    0,                                        // tp_weaklistoffset
    0,                                        // tp_iter
    0,                                        // tp_iternext
    0,                                        // tp_methods
    0,                                        // tp_members
    rational_getset,                          // tp_getset
    0,                                        // tp_base
    0,                                        // tp_dict
    0,                                        // tp_descr_get
    0,                                        // tp_descr_set
    0,                                        // tp_dictoffset
    0,                                        // tp_init
    0,                                        // tp_alloc
    rational_new,                             // tp_new
    0,                                        // tp_free
};

// Numpy support

PyObject* rational_getitem(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(rational));
    return PyRational_FromRational(r);
}

int rational_setitem(PyObject* item, void* data, void* arr) {
    rational r;
    if (PyRational_Check(item))
        r = ((PyRational*)item)->r;
    else {
        long n = PyInt_AsLong(item);
        if (n==-1 && PyErr_Occurred())
            return -1;
        PyObject* y = PyInt_FromLong(n);
        if (!y)
            return -1;
        int eq = PyObject_RichCompareBool(item,y,Py_EQ);
        Py_DECREF(y);
        if (eq<0)
            return -1;
        if (!eq) {
            PyErr_Format(PyExc_TypeError,"expected rational, got %s",item->ob_type->tp_name);
            return -1;
        }
        r = n;
    }
    memcpy(data,&r,sizeof(rational));
    return 0;
}

template<class T> inline void byteswap(T& x) {
    char* p = (char*)&x;
    for (size_t i = 0; i < sizeof(T)/2; i++)
        swap(p[i],p[sizeof(T)-1-i]);
}

void rational_copyswapn(void* dst_, npy_intp dstride, void* src_, npy_intp sstride, npy_intp n, int swap, void* arr) {
    char *dst = (char*)dst_, *src = (char*)src_;
    if (!src)
        return;
    if (swap)
        for (npy_intp i = 0; i < n; i++) {
            rational& r = *(rational*)(dst+dstride*i);
            memcpy(&r,src+sstride*i,sizeof(rational));
            byteswap(r.n);
            byteswap(r.dmm);
        }
    else if (dstride==sizeof(rational) && sstride==sizeof(rational))
        memcpy(dst,src,n*sizeof(rational));
    else
        for (npy_intp i = 0; i < n; i++)
            memcpy(dst+dstride*i,src+sstride*i,sizeof(rational));
}

void rational_copyswap(void* dst, void* src, int swap, void* arr) {
    if (!src)
        return;
    rational& r = *(rational*)dst;
    memcpy(&r,src,sizeof(rational));
    if (swap) {
        byteswap(r.n);
        byteswap(r.dmm);
    }
}

int rational_compare(const void* d0, const void* d1, void* arr) {
    rational x = *(rational*)d0,
             y = *(rational*)d1;
    return x<y?-1:x==y?0:1;
}

#define FIND_EXTREME(name,op) \
    int rational_##name(void* data_, npy_intp n, npy_intp* max_ind, void* arr) { \
        if (!n) \
            return 0; \
        const rational* data = (rational*)data_; \
        npy_intp best_i = 0; \
        rational best_r = data[0]; \
        for (npy_intp i = 1; i < n; i++) \
            if (data[i] op best_r) { \
                best_i = i; \
                best_r = data[i]; \
            } \
        *max_ind = best_i; \
        return 0; \
    }
FIND_EXTREME(argmin,<)
FIND_EXTREME(argmax,>)

void rational_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1, void* op, npy_intp n, void* arr) {
    rational r;
    try {
        const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
        for (npy_intp i = 0; i < n; i++) {
            r = r + (*(rational*)ip0)*(*(rational*)ip1);
            ip0 += is0;
            ip1 += is1;
        }
        *(rational*)op = r;
    } catch (const exception& e) {
        set_python_error(e);
    }
}

npy_bool rational_nonzero(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(r));
    return r?NPY_TRUE:NPY_FALSE;
}

int rational_fill(void* data_, npy_intp length, void* arr) {
    rational* data = (rational*)data_;
    try {
        rational delta = data[1]-data[0];
        rational r = data[1];
        for (npy_intp i = 2; i < length; i++) {
            r = r+delta;
            data[i] = r;
        }
    } catch (const exception& e) {
        set_python_error(e);
    }
    return 0;
}

int rational_fillwithscalar(void* buffer_, npy_intp length, void* value, void* arr) {
    rational r = *(rational*)value;
    rational* buffer = (rational*)buffer_;
    for (npy_intp i = 0; i < length; i++)
        buffer[i] = r;
    return 0;
}

PyArray_ArrFuncs rational_arrfuncs;

struct align_test { char c; struct {int64_t i[2];} r; };

PyArray_Descr rational_descr = {
    PyObject_HEAD_INIT(0)
    &PyRational_Type,       // typeobj
    'V',                    // kind
    'r',                    // type
    '=',                    // byteorder
    // For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our exceptions.  This isn't technically necessary,
    // since we're careful about thread safety, and hopefully future versions of numpy will recognize that.
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, // hasobject
    0,                      // type_num
    sizeof(rational),       // elsize
    offsetof(align_test,r), // alignment
    0,                      // subarray
    0,                      // fields
    0,                      // names
    &rational_arrfuncs,     // f
};

template<class From,class To> void numpy_cast(void* from_, void* to_, npy_intp n, void* fromarr, void* toarr) {
    const From* from = (From*)from_;
    To* to = (To*)to_;
    try {
        for (npy_intp i = 0; i < n; i++)
            to[i] = cast<To>(from[i]);
    } catch (const exception& e) {
        set_python_error(e);
    }
}

template<class From,class To> int register_cast(PyArray_Descr* from_descr, int to_typenum, bool safe) {
    int r = PyArray_RegisterCastFunc(from_descr,to_typenum,numpy_cast<From,To>);
    if (!r && safe)
        r = PyArray_RegisterCanCast(from_descr,to_typenum,NPY_NOSCALAR);
    return r;
}

#define BINARY_UFUNC(name,intype0,intype1,outtype,exp) \
    void name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
        npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions; \
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        try { \
            for (int k = 0; k < n; k++) { \
                intype0 x = *(intype0*)i0; \
                intype1 y = *(intype1*)i1; \
                *(outtype*)o = exp; \
                i0 += is0; i1 += is1; o += os; \
            } \
        } catch (const exception& e) { \
            set_python_error(e); \
        } \
    }
#define RATIONAL_BINARY_UFUNC(name,type,exp) BINARY_UFUNC(rational_ufunc_##name,rational,rational,type,exp)
RATIONAL_BINARY_UFUNC(add,rational,x+y)
RATIONAL_BINARY_UFUNC(subtract,rational,x-y)
RATIONAL_BINARY_UFUNC(multiply,rational,x*y)
RATIONAL_BINARY_UFUNC(divide,rational,x/y)
RATIONAL_BINARY_UFUNC(remainder,rational,x%y)
RATIONAL_BINARY_UFUNC(floor_divide,rational,floor(x/y))
PyUFuncGenericFunction rational_ufunc_true_divide = rational_ufunc_divide;
RATIONAL_BINARY_UFUNC(minimum,rational,min(x,y))
RATIONAL_BINARY_UFUNC(maximum,rational,max(x,y))
RATIONAL_BINARY_UFUNC(equal,bool,x==y)
RATIONAL_BINARY_UFUNC(not_equal,bool,x!=y)
RATIONAL_BINARY_UFUNC(less,bool,x<y)
RATIONAL_BINARY_UFUNC(greater,bool,x>y)
RATIONAL_BINARY_UFUNC(less_equal,bool,x<=y)
RATIONAL_BINARY_UFUNC(greater_equal,bool,x>=y)

BINARY_UFUNC(gcd_ufunc,int64_t,int64_t,int64_t,gcd(x,y))
BINARY_UFUNC(lcm_ufunc,int64_t,int64_t,int64_t,lcm(x,y))

#define UNARY_UFUNC(name,type,exp) \
    void rational_ufunc_##name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
        npy_intp is = steps[0], os = steps[1], n = *dimensions; \
        char *i = args[0], *o = args[1]; \
        try { \
            for (int k = 0; k < n; k++) { \
                rational x = *(rational*)i; \
                *(type*)o = exp; \
                i += is; o += os; \
            } \
        } catch (const exception& e) { \
            set_python_error(e); \
        } \
    }
UNARY_UFUNC(negative,rational,-x)
UNARY_UFUNC(absolute,rational,abs(x))
UNARY_UFUNC(floor,rational,floor(x))
UNARY_UFUNC(ceil,rational,ceil(x))
UNARY_UFUNC(trunc,rational,cast<long>(x))
UNARY_UFUNC(square,rational,x*x)
UNARY_UFUNC(rint,rational,rint(x))
UNARY_UFUNC(sign,rational,sign(x))
UNARY_UFUNC(reciprocal,rational,inverse(x))
UNARY_UFUNC(numerator,int64_t,x.n)
UNARY_UFUNC(denominator,int64_t,x.d())

PyMethodDef module_methods[] = {
    {0} // sentinel
};

}

PyMODINIT_FUNC
initrational() {
    // Initialize numpy
    import_array();
    if (PyErr_Occurred()) return;
    import_umath();
    if (PyErr_Occurred()) return;
    PyObject* numpy_str = PyString_FromString("numpy");
    if (!numpy_str) return;
    PyObject* numpy = PyImport_Import(numpy_str);
    Py_DECREF(numpy_str);
    if (!numpy) return;

    // Can't set this until we import numpy
    PyRational_Type.tp_base = &PyGenericArrType_Type;

    // Initialize rational type object
    if (PyType_Ready(&PyRational_Type) < 0)
        return;

    // Initialize rational descriptor
    PyArray_InitArrFuncs(&rational_arrfuncs);
    rational_arrfuncs.getitem = rational_getitem;
    rational_arrfuncs.setitem = rational_setitem;
    rational_arrfuncs.copyswapn = rational_copyswapn;
    rational_arrfuncs.copyswap = rational_copyswap;
    rational_arrfuncs.compare = rational_compare;
    rational_arrfuncs.argmin = rational_argmin;
    rational_arrfuncs.argmax = rational_argmax;
    rational_arrfuncs.dotfunc = rational_dot;
    rational_arrfuncs.nonzero = rational_nonzero;
    rational_arrfuncs.fill = rational_fill;
    rational_arrfuncs.fillwithscalar = rational_fillwithscalar;
    // Left undefined: scanfunc, fromstr, sort, argsort
    rational_descr.ob_type = &PyArrayDescr_Type;
    int npy_rational = PyArray_RegisterDataType(&rational_descr);
    if (npy_rational<0) return;

    // Support dtype(rational) syntax
    if (PyDict_SetItemString(PyRational_Type.tp_dict,"dtype",(PyObject*)&rational_descr)<0) return;

    // Register casts to and from rational
    #define REGISTER_INT_CONVERSIONS(bits) \
        if (register_cast<int##bits##_t,rational>(PyArray_DescrFromType(NPY_INT##bits),npy_rational,true)<0) return; \
        if (register_cast<uint##bits##_t,rational>(PyArray_DescrFromType(NPY_UINT##bits),npy_rational,true)<0) return; \
        if (register_cast<rational,int##bits##_t>(&rational_descr,NPY_INT##bits,false)<0) return; \
        if (register_cast<rational,uint##bits##_t>(&rational_descr,NPY_UINT##bits,false)<0) return;
    REGISTER_INT_CONVERSIONS(8)
    REGISTER_INT_CONVERSIONS(16)
    REGISTER_INT_CONVERSIONS(32)
    REGISTER_INT_CONVERSIONS(64)
    if (register_cast<rational,float >(&rational_descr,NPY_FLOAT,false)<0) return;
    if (register_cast<rational,double>(&rational_descr,NPY_DOUBLE,true)<0) return;
    if (register_cast<bool,rational>(PyArray_DescrFromType(NPY_BOOL),npy_rational,true)<0) return;
    if (register_cast<rational,bool>(&rational_descr,NPY_BOOL,false)<0) return;

    // Register ufuncs
    #define REGISTER_UFUNC(name,...) ({ \
        PyUFuncObject* ufunc = (PyUFuncObject*)PyObject_GetAttrString(numpy,#name); \
        if (!ufunc) return; \
        int _types[] = __VA_ARGS__; \
        if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
            PyErr_Format(PyExc_AssertionError,"ufunc %s takes %d arguments, our loop takes %ld",#name,ufunc->nargs,sizeof(_types)/sizeof(int)); \
            return; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_##name,_types,0)<0) return; \
        });
    #define REGISTER_UFUNC_BINARY_RATIONAL(name) REGISTER_UFUNC(name,{npy_rational,npy_rational,npy_rational})
    #define REGISTER_UFUNC_BINARY_COMPARE(name) REGISTER_UFUNC(name,{npy_rational,npy_rational,NPY_BOOL})
    #define REGISTER_UFUNC_UNARY(name) REGISTER_UFUNC(name,{npy_rational,npy_rational})
    // Binary
    REGISTER_UFUNC_BINARY_RATIONAL(add)
    REGISTER_UFUNC_BINARY_RATIONAL(subtract)
    REGISTER_UFUNC_BINARY_RATIONAL(multiply)
    REGISTER_UFUNC_BINARY_RATIONAL(divide)
    REGISTER_UFUNC_BINARY_RATIONAL(remainder)
    REGISTER_UFUNC_BINARY_RATIONAL(true_divide)
    REGISTER_UFUNC_BINARY_RATIONAL(floor_divide)
    REGISTER_UFUNC_BINARY_RATIONAL(minimum)
    REGISTER_UFUNC_BINARY_RATIONAL(maximum)
    // Comparisons
    REGISTER_UFUNC_BINARY_COMPARE(equal)
    REGISTER_UFUNC_BINARY_COMPARE(not_equal)
    REGISTER_UFUNC_BINARY_COMPARE(less)
    REGISTER_UFUNC_BINARY_COMPARE(greater)
    REGISTER_UFUNC_BINARY_COMPARE(less_equal)
    REGISTER_UFUNC_BINARY_COMPARE(greater_equal)
    // Unary
    REGISTER_UFUNC_UNARY(negative)
    REGISTER_UFUNC_UNARY(absolute)
    REGISTER_UFUNC_UNARY(floor)
    REGISTER_UFUNC_UNARY(ceil)
    REGISTER_UFUNC_UNARY(trunc)
    REGISTER_UFUNC_UNARY(rint)
    REGISTER_UFUNC_UNARY(square)
    REGISTER_UFUNC_UNARY(reciprocal)
    REGISTER_UFUNC_UNARY(sign)

    // Create module
    PyObject* m = Py_InitModule3("rational", module_methods,
        "Fixed precision rational numbers, including numpy support");
    if (!m) return;

    // Add rational type
    Py_INCREF(&PyRational_Type);
    PyModule_AddObject(m,"rational",(PyObject*)&PyRational_Type);

    // Create numerator and denominator ufuncs
    #define NEW_UNARY_UFUNC(name,type,doc) ({ \
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1,PyUFunc_None,(char*)#name,(char*)doc,0); \
        if (!ufunc) return; \
        int types[2] = {npy_rational,type}; \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_##name,types,0)<0) return; \
        PyModule_AddObject(m,#name,(PyObject*)ufunc); \
        })
    NEW_UNARY_UFUNC(numerator,NPY_INT64,"rational number numerator");
    NEW_UNARY_UFUNC(denominator,NPY_INT64,"rational number denominator");

    // Create gcd and lcm ufuncs
    #define GCD_LCM_UFUNC(name,type,doc) ({ \
        static const PyUFuncGenericFunction func[1] = {name##_ufunc}; \
        static const char types[3] = {type,type,type}; \
        static void* data[1] = {0}; \
        PyObject* ufunc = PyUFunc_FromFuncAndData((PyUFuncGenericFunction*)func,data,(char*)types,1,2,1,PyUFunc_One,(char*)#name,(char*)doc,0); \
        if (!ufunc) return; \
        PyModule_AddObject(m,#name,(PyObject*)ufunc); \
        })
    GCD_LCM_UFUNC(gcd,NPY_INT64,"greatest common denominator of two integers");
    GCD_LCM_UFUNC(lcm,NPY_INT64,"least common multiple of two integers");
}
