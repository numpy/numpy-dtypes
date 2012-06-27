Creating New Generalized UFunc for Custom DType
-----------------------------------------------

.. highlight:: c

The next example shows how to create a new generalized ufunc for the Rational dtype.
The gufunc example creates a gufunc 'matrix_multiply' which loops over a pair of 
vectors or matrices and performs a matrix multiply on each pair of matrix elements.


1.  A loop function is created to loop through the outer or loop dimensions, performing a
    matrix multiply operation on the core dimensions for each loop::
 
        static void
        rational_gufunc_matrix_multiply(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func))
        {
            // outer dimensions counter
            npy_intp N_;

            // length of flattened outer dimensions
            npy_intp dN = dimensions[0];

            // striding over flattened outer dimensions for input and output arrays
            npy_intp s0 = steps[0];
            npy_intp s1 = steps[1];
            npy_intp s2 = steps[2];

            // loop through outer dimensions, performing matrix multiply on core dimensions for each loop
            for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
                rational_matrix_multiply(args, dimensions+1, steps+3);
            }
        }

    If the input matrices have more than one outer dimension, the outer dimensions are flattened from the
    perspective of the loop function.

    The matrix multiply function::

        static NPY_INLINE void
        rational_matrix_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
        {
            // pointers to data for input and output arrays
            char *ip1 = args[0];
            char *ip2 = args[1];
            char *op = args[2];

            // lengths of core dimensions
            npy_intp dm = dimensions[0];
            npy_intp dn = dimensions[1];
            npy_intp dp = dimensions[2];

            // striding over core dimensions
            npy_intp is1_m = steps[0];
            npy_intp is1_n = steps[1];
            npy_intp is2_n = steps[2];
            npy_intp is2_p = steps[3];
            npy_intp os_m = steps[4];
            npy_intp os_p = steps[5];

            // core dimensions counters
            npy_intp m, n, p;

            // calculate dot product for each row/column vector pair
            for (m = 0; m < dm; m++) {
                for (p = 0; p < dp; p++) {
                    npyrational_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

                    ip2 += is2_p;
                    op  +=  os_p;
                }

                ip2 -= is2_p * p;
                op -= os_p * p;

                ip1 += is1_m;
                op += os_m;
            }
        }

    |

2.  In the 'initrational' function used to initialize the Rational dtype with numpy,
    a new PyUFuncObject is created for the new 'matrix_multiply' generalized ufunc using the
    PyUFunc_FromFuncAndDataAndSignature function::

        PyObject* gufunc = PyUFunc_FromFuncAndDataAndSignature(0,0,0,0,2,1,PyUFunc_None,(char*)"matrix_multiply",(char*)"return result of multiplying two matrices of rationals",0,"(m,n),(n,p)->(m,p)");

    This is identical to the PyUFunc_FromFuncAndData function used to create a ufunc object in the examples above,
    with the addition of a ufunc signature describing the core dimensions of the input and output arrays. In this
    example, the generalized ufunc operates on pairs of matrices with dimensions (m,n) and (n,p), producing an
    output matrix of dimensions (m,p).

3.  The loop function is registered using the loop function and the PyUFuncObject created in step 2::

        int types2[3] = {npy_rational,npy_rational,npy_rational};
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)gufunc,npy_rational,rational_gufunc_matrix_multiply,types2,0) < 0) {
            return;
        }

4.  Finally, a function called 'matrix_multiply' is added to the rational module which
    will call the numerator ufunc::

        PyModule_AddObject(m,"matrix_multiply",(PyObject*)gufunc);

    |

An example of using the add ufunc with the Rational dtype::

    In [1]: import numpy as np

    In [2]: from rational import rational, matrix_multiply

    In [3]: r1=rational(1,2)

    In [4]: r2=rational(3,4)

    In [5]: r3=rational(5,6)

    In [6]: r4=rational(7,8)

    In [7]: a=np.array([[[[r1,r2],[r3,r4]],[[r1,r2],[r3,r4]]], [[[r1,r2],[r3,r4]],[[r1,r2],[r3,r4]]]], dtype=rational)

    In [8]: b=np.array([[[[r3,r4],[r1,r2]],[[r3,r4],[r1,r2]]], [[[r3,r4],[r1,r2]],[[r3,r4],[r1,r2]]]], dtype=rational)

    In [9]: matrix_multiply(a,b)
    Out[9]: 
    array([[[[19/24, 1],
             [163/144, 133/96]],

            [[19/24, 1],
             [163/144, 133/96]]],


           [[[19/24, 1],
             [163/144, 133/96]],

            [[19/24, 1],
             [163/144, 133/96]]]], dtype=rational)

