Extending Existing UFunc for Custom DType
-----------------------------------------

.. highlight:: c

The first example shows how to extend the existing 'add' ufunc for the Rational
dtype. The add ufunc is extended for Rational dtypes using Rational's
'rational_add' function which takes two rational numbers and returns a rational
object representing the sum of those two rational numbers::

    static NPY_INLINE rational
    rational_add(rational x, rational y) {
        // d(y) retrieves denominator for y
        return make_rational_fast((int64_t)x.n*d(y) + (int64_t)d(x)*y.n, (int64_t)d(x)*d(y));
    }

1.  A 1-d loop function is created which loops over each pair of elements from two
    1-d arrays of rationals and calls rational_add for each pair::

        void rational_ufunc_add(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
            npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
            char *i0 = args[0], *i1 = args[1], *o = args[2];
            int k;
            for (k = 0; k < n; k++) {
                rational x = *(rational*)i0;
                rational y = *(rational*)i1;
                *(rational*)o = rational_add(x,y);
                i0 += is0; i1 += is1; o += os;
            }
        }

    The loop function must have the exact signature as above. The function parameters are:

    - char \**args - array of pointers to the actual data for the input and output arrays.
      In this example there are three pointers: two pointers pointing to blocks of
      memory for the two input arrays, and one pointer pointing to the block of
      memory for the output array. The result of rational_add should be stored in
      the output array.

    - dimensions - a pointer to the size of the dimension over which this function is looping

    - steps - a pointer to the number of bytes to jump to get to the next element in this
      dimension for each of the input and output arguments

    - data - arbitrary data (extra arguments, function names, etc.) that can be stored with
      the ufunc and will be passed in when it is called

    The Rational dtype has an example of a C macro which can be used to generate create the
    above function for different rational ufuncs::

        #define BINARY_UFUNC(name,intype0,intype1,outtype,exp) \
            void name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
                npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions; \
                char *i0 = args[0], *i1 = args[1], *o = args[2]; \
                int k; \
                for (k = 0; k < n; k++) { \
                    intype0 x = *(intype0*)i0; \
                    intype1 y = *(intype1*)i1; \
                    *(outtype*)o = exp; \
                    i0 += is0; i1 += is1; o += os; \
                } \
            }

        #define RATIONAL_BINARY_UFUNC(name, type, exp) BINARY_UFUNC(rational_ufunc_##name, rational, rational, type, exp)

    which can be used like so::

        RATIONAL_BINARY_UFUNC(add, rational, rational_add(x,y))

    with the following arguments:

    - name suffix of 1-d loop function (the generated loop function will have the name 'rational_ufunc_<name>'
    - output type
    - expression to calculate the output value for each pair of input elements.
      In this example the expression is a call to the function rational_add.

    |

2.  In the 'initrational' function used to initialize the Rational dtype with numpy, a
    PyUFuncObject is obtained for the existing 'add' ufunc in the numpy module::

        PyUFuncObject* ufunc = (PyUFuncObject*)PyObject_GetAttrString(numpy,"add");

    |

3.  The 1-d loop function is registered using the PyUFuncObject obtained in step 2::

        int types[] = {npy_rational,npy_rational,npy_rational};

        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_add,types,0) < 0) {
            return;
        }

    The function parameters are:

    - pointer to PyUFuncObject obtained in step 2
    - custom rational dtype id (obtained when dtype is registered with call to PyArray_RegisterDataType)
    - 1-d loop function
    - array of input and output type ids (in this case two input rational types and one
      output rational type)
    - pointer to arbitrary data that will be passed to 1-d loop function

    |

4.  Steps 2-3 can also be accomplished by defining a c macro similar to the one
    provided with Rational::

        #define REGISTER_UFUNC(name,...) { \
            PyUFuncObject* ufunc = (PyUFuncObject*)PyObject_GetAttrString(numpy,#name); \
            if (!ufunc) { \
                return; \
            } \
            int _types[] = __VA_ARGS__; \
            if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
                PyErr_Format(PyExc_AssertionError,"ufunc %s takes %d arguments, our loop takes %ld",#name,ufunc->nargs,sizeof(_types)/sizeof(int)); \
                return; \
            } \
            if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_##name,_types,0)<0) { \
                return; \
            } \
        }
        #define REGISTER_UFUNC_BINARY_RATIONAL(name) REGISTER_UFUNC(name,{npy_rational,npy_rational,npy_rational})
    
    and calling it like so::

        REGISTER_UFUNC_BINARY_RATIONAL(add)

    |

An example of using the add ufunc with the Rational dtype::

    In [1]: import numpy as np

    In [2]: from rational import rational

    In [3]: r1=rational(1,2)

    In [4]: r2=rational(3,4)

    In [5]: r3=rational(5,6)

    In [6]: r4=rational(7,8)

    In [7]: a=np.array([r1,r2], dtype=rational)

    In [8]: b=np.array([r3,r4], dtype=rational)

    In [9]: np.add(a,b)
    Out[9]: array([4/3, 13/8], dtype=rational)

