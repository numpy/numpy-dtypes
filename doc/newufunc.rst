Creating New UFunc for Custom DType
-----------------------------------

.. highlight:: c

The next example shows how to create a new ufunc for the Rational dtype. The ufunc
example creates a ufunc called 'numerator' which generates an array of numerator
values based rational numbers from the input array. 

1.  A 1-d loop function is created as before which takes the numerator value from
    each element of the input array and stores it in the output array::

        void rational_ufunc_numerator(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
            npy_intp is = steps[0], os = steps[1], n = *dimensions;
            char *i = args[0], *o = args[1];
            int k;
            for (k = 0; k < n; k++) {
                rational x = *(rational*)i;
                *(int64_t*)o = x.n;
                i += is; o += os;
            }
        }

    You can also use the c macro provided in Rational for generating the above function::

        #define UNARY_UFUNC(name,type,exp) \
            void rational_ufunc_##name(char** args, npy_intp* dimensions, npy_intp* steps, void* data) { \
                npy_intp is = steps[0], os = steps[1], n = *dimensions; \
                char *i = args[0], *o = args[1]; \
                int k; \
                for (k = 0; k < n; k++) { \
                    rational x = *(rational*)i; \
                    *(type*)o = exp; \
                    i += is; o += os; \
                } \
            }

    and call it like so::

        UNARY_UFUNC(numerator,int64_t,x.n)

    |

2.  In the 'initrational' function used to initialize the Rational dtype with numpy,
    a new PyUFuncObject is created for the new 'numerator' ufunc using the
    PyUFunc_FromFuncAndData function::

        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1,PyUFunc_None,(char*)"numerator",(char*)"rational number numerator",0);

    In this use case, the first four parameters should be set to zero since we're
    creating a new ufunc without support for existing dtypes. The rest of the
    parameters:

    - number of inputs to function that the loop function calls for each pair of elements
    - number of outputs of loop function
    - name of the ufunc
    - documentation string describing the ufunc
    - unused; present for backwards compatibility

    |

3.  The 1-d loop function is registered using the loop function and the
    PyUFuncObject created in step 2::

        int _types[] = {npy_rational,NPY_INT64};

        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_numerator,_types,0)<0) {
            return;
        }

    |

4.  Finally, a function called 'numerator' is added to the rational module which
    will call the numerator ufunc::

        PyModule_AddObject(m,"numerator",(PyObject*)ufunc);

    |

5.  Steps 2-4 can also be accomplished by using a c macro similar to the one
    provided with Rational::

        #define NEW_UNARY_UFUNC(name,type,doc) { \
            PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1,PyUFunc_None,(char*)#name,(char*)doc,0); \
            if (!ufunc) { \
                return; \
            } \
            int types[2] = {npy_rational,type}; \
            if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,npy_rational,rational_ufunc_##name,types,0)<0) { \
                return; \
            } \
            PyModule_AddObject(m,#name,(PyObject*)ufunc); \
        }

    and calling it like so::

        NEW_UNARY_UFUNC(numerator,NPY_INT64,"rational number numerator");

    |

An example of using the numerator ufunc with the Rational dtype::

    In [1]: import numpy as np

    In [2]: from rational import rational, numerator

    In [3]: r1=rational(1,2)

    In [4]: r2=rational(3,4)

    In [5]: r3=rational(5,6)

    In [6]: r4=rational(7,8)

    In [7]: a=np.array([r1,r2,r3,r4], dtype=rational)

    In [8]: numerator(a)
    Out[8]: array([1, 3, 5, 7])

