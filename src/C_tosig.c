
/* A file to test importing C modules for handling arrays to Python */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
//#include "arrayobject.h" //NUMPY
#include <numpy/arrayobject.h>

#include "C_tosig.h"
#include <math.h>
#include "ToSig.h"

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_tosigMethods[] = {
		{"stream2logsig", tologsig, METH_VARARGS,"stream2logsig(array(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, \"the data in stream space\" and returns a numpy vector containing the logsignature of the vector series up to given signature_degree"},
		{"stream2sig", tosig, METH_VARARGS, "stream2logsig(array(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, \"the data in stream space\" and returns a numpy vector containing the signature of the vector series up to given signature_degree"},
		{"logsigdim", getlogsigsize, METH_VARARGS,"logsigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the dimension of the log signature vector returned by array2logsig"},
		{"sigdim", getsigsize, METH_VARARGS,"sigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the length of the signature vector returned by array2logsig"},
		{"logsigkeys",showlogsigkeys, METH_VARARGS, "logsigkeys(signal_dimension, signature_degree) returns, in the order used by ...2logsig, a space separated ascii string containing the keys associated the entries in the log signature returned by ...2logsig"},
		{"sigkeys",showsigkeys, METH_VARARGS, "sigkeys(signal_dimension, signature_degree) returns, in the order used by ...2sig, a space separated ascii string containing the keys associated the entries in the signature returned by ...2sig"},
		{"retrieve_capsule", retrieveCapsule, METH_VARARGS, "Some description of retrieve_capsule()"},
		{NULL, NULL, 0, NULL}        /* Sentinel */
};


/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_tosig in compile and linked 

//static PyObject *tosigerror;
#if PY_MAJOR_VERSION < 3
//PyMODINIT_FUNC // different meanings in different levels of python // for python 2.7:
void
inittosig(void)
{
	PyObject *m;
	m = Py_InitModule("tosig", _C_tosigMethods);
	if (m == NULL) return;
	
	import_array();  // Must be present for NumPy.  Called first after above line.

	//tosigerror = PyErr_NewException("tosig.error", NULL, NULL);
	//Py_INCREF(tosigerror);
	//PyModule_AddObject(m, "error", tosigerror);
}
#else
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
    "tosig",     /* m_name */
    "This is the tosig module from ESIG",  /* m_doc */
    -1,                  /* m_size */
    _C_tosigMethods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
    };

PyMODINIT_FUNC
PyInit_tosig(void){
  import_array();
  return PyModule_Create(&moduledef);
}
#endif
/* ==== Operate on Matrix as a vector time series returning a vectorlog signature ==

    Returns a NEW NumPy vector
    interface:  tologsig(series1, depth)
                series1 is NumPy matrix
				depth is a positive integer of size_t
                returns a NumPy vector                                       */

static PyObject* tologsig(PyObject* self, PyObject* args)
{
	PyArrayObject *seriesin, *vecout;
	//double *cout;
	//Py_ssize_t width, depth, recs;
	Py_ssize_t depth;
	npy_intp width;
	npy_intp dims[2];

	/* Parse tuple */
	if (!PyArg_ParseTuple(args, "O!n",
		&PyArray_Type, &seriesin, &depth))  return NULL;
	if (NULL == seriesin)  return NULL;

	/* Check that object input is 'double' type and a matrix*/
	//if (not_valid_matrix(seriesin)) return NULL;

	/* Get the dimensions of the input */
	//width = seriesin->dimensions[1];
	//recs = seriesin->dimensions[0];
	width = PyArray_DIM(seriesin, 1);
	dims[0] = (npy_intp) GetLogSigSize((size_t)width, (size_t)depth);

	/* Make a new double matrix of same dims */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	/* Do the calculation. */
	GetLogSig(seriesin, vecout, width, depth);

	return PyArray_Return(vecout);
}

/* ==== Operate on Matrix as a vector time series returning a vector signature ==

    Returns a NEW NumPy vector
    interface:  tosig(series1, depth)
                series1 is NumPy matrix
				depth is a positive integer of Py_ssize_t
                returns a NumPy vector                                       */

static PyObject* tosig(PyObject* self, PyObject* args)
{
	PyArrayObject *seriesin, *vecout;
	//double *cout;
	//Py_ssize_t width, depth, recs;
	Py_ssize_t depth;
	npy_intp width;
	npy_intp dims[2];

	/* Parse tuple */
	if (!PyArg_ParseTuple(args, "O!n",
		&PyArray_Type, &seriesin, &depth))  return NULL;
	if (NULL == seriesin)  return NULL;

	/* Check that object input is 'double' type and a matrix*/
	//if (not_valid_matrix(seriesin)) return NULL;

	/* Get the dimensions of the input */
	//width = seriesin->dimensions[1];
	//recs = seriesin->dimensions[0];
	width = PyArray_DIM(seriesin, 1);
	GetLogSigSize((size_t)width, (size_t)depth); //initialise basis
	dims[0] = (npy_intp) GetSigSize((size_t)width, (size_t)depth);

	/* Make a new double vector of correct dimension */
	vecout=(PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	/* Do the calculation. */
	GetSig(seriesin, vecout, width, depth);

	return PyArray_Return(vecout);
}

static PyObject* retrieveCapsule(PyObject* self, PyObject* args)
{
	PyArrayObject *seriesin, *vecout;
	//double *cout;
	//Py_ssize_t width, depth, recs;
	Py_ssize_t depth;
	npy_intp width;
	npy_intp dims[2];

	/* Parse tuple */
	if (!PyArg_ParseTuple(args, "O!n",
		&PyArray_Type, &seriesin, &depth))  return NULL;
	if (NULL == seriesin)  return NULL;

	/* Check that object input is 'double' type and a matrix*/
	//if (not_valid_matrix(seriesin)) return NULL;

	/* Get the dimensions of the input */
	//width = seriesin->dimensions[1];
	//recs = seriesin->dimensions[0];
	width = PyArray_DIM(seriesin, 1);
	GetLogSigSize((size_t)width, (size_t)depth); //initialise basis
	dims[0] = (npy_intp)GetSigSize((size_t)width, (size_t)depth);

	/* Make a new double vector of correct dimension */
	vecout = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	/* Do the calculation. */
	//GetSig(seriesin, vecout, width, depth);

	return PyArray_Return(vecout);
}

/* ==== Determines the size of log signature =========================
    Returns a NEW  NumPy vector array
    interface:  getlogsigsize(width,depth)
                width and depth are Py_ssize_t and are at least 2
                returns a Py_ssize_t                                       */
static PyObject* getlogsigsize(PyObject* self, PyObject* args)
{
	Py_ssize_t depth, width, ans;

	/* Parse tuple */
	if (!PyArg_ParseTuple(args, "nn",
		&width, &depth))  return NULL;

	ans = GetLogSigSize((size_t)width, (size_t)depth);
	return Py_BuildValue("n", ans);
}

/* ==== Determines the size of  signature =========================
    Returns a NEW  NumPy vector array
    interface:  getsigsize(width,depth)
                width and depth are Py_ssize_t and are at least 2
                returns a Py_ssize_t                                       */
static PyObject* getsigsize(PyObject* self, PyObject* args)
{
	Py_ssize_t depth, width, ans;

	/* Parse tuple */
	if (!PyArg_ParseTuple(args, "nn",
		&width, &depth))  return NULL;

	ans = GetSigSize((size_t)width, (size_t)depth);
	return Py_BuildValue("n", ans);
}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */
int not_valid_matrix(PyArrayObject* mat)
{
  if (PyArray_TYPE(mat) != NPY_DOUBLE || PyArray_NDIM(mat) != 2) {
		PyErr_SetString(PyExc_ValueError,
			"In not_valid_matrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}
/*
int not_doublematrix(PyArrayObject* mat)
{
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2) {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}
*/

///* #### Vector Extensions ############################## */
//
///* ==== vector function - manipulate vector in place ======================
//    Multiply the input by 2 x dfac and put in output
//    Interface:  vecfcn1(vec1, vec2, str1, d1)
//                vec1, vec2 are NumPy vectors, 
//                str1 is Python string, d1 is Python float (double)
//                Returns integer 1 if successful                */
//static PyObject* vecfcn1(PyObject* self, PyObject* args)
//{
//	PyArrayObject* vecin, *vecout;  // The python objects to be extracted from the args
//	double* cin, *cout;             // The C vectors to be created to point to the 
//	//   python vectors, cin and cout point to the row
//	//   of vecin and vecout, respectively
//	int i, j, n;
//	const char* str;
//	double dfac;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!O!sd", &PyArray_Type, &vecin,
//		&PyArray_Type, &vecout, &str, &dfac))  return NULL;
//	if (NULL == vecin)  return NULL;
//	if (NULL == vecout)  return NULL;
//
//	// Print out input string
//	printf("Input string: %s\n", str);
//
//	/* Check that objects are 'double' type and vectors
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublevector(vecin)) return NULL;
//	if (not_doublevector(vecout)) return NULL;
//
//	/* Change contiguous arrays into C * arrays   */
//	cin=pyvector_to_Carrayptrs(vecin);
//	cout=pyvector_to_Carrayptrs(vecout);
//
//	/* Get vector dimension. */
//	n=vecin->dimensions[0];
//
//	/* Operate on the vectors  */
//	for (i=0; i<n; i++) {
//		cout[i]=2.0*dfac*cin[i];
//	}
//
//	return Py_BuildValue("i", 1);
//}
//
///* ==== Square vector components & multiply by a float =========================
//    Returns a NEW  NumPy vector array
//    interface:  vecsq(vec1, x1)
//                vec1 is NumPy vector, x1 is Python float (double)
//                returns a NumPy vector                                        */
//static PyObject* vecsq(PyObject* self, PyObject* args)
//{
//	PyArrayObject* vecin, *vecout;
//	double* cin, *cout, dfactor;   // The C vectors to be created to point to the 
//	//   python vectors, cin and cout point to the row
//	//   of vecin and vecout, respectively
//	int i, j, n, m, dims[2];
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!d",
//		&PyArray_Type, &vecin, &dfactor))  return NULL;
//	if (NULL == vecin)  return NULL;
//
//	/* Check that object input is 'double' type and a vector
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublevector(vecin)) return NULL;
//
//	/* Get the dimension of the input */
//	n=dims[0]=vecin->dimensions[0];
//
//	/* Make a new double vector of same dimension */
//	vecout=(PyArrayObject*) PyArray_FromDims(1, dims, NPY_DOUBLE);
//
//	/* Change contiguous arrays into C *arrays   */
//	cin=pyvector_to_Carrayptrs(vecin);
//	cout=pyvector_to_Carrayptrs(vecout);
//
//	/* Do the calculation. */
//	for (i=0; i<n; i++) {
//		cout[i]= dfactor*cin[i]*cin[i];
//	}
//
//	return PyArray_Return(vecout);
//}
//
///* #### Vector Utility functions ######################### */
//
///* ==== Make a Python Array Obj. from a PyObject, ================
//     generates a double vector w/ contiguous memory which may be a new allocation if
//     the original was not a double type or contiguous 
//  !! Must DECREF the object returned from this routine unless it is returned to the
//     caller of this routines caller using return PyArray_Return(obj) or
//     PyArray_BuildValue with the "N" construct   !!!
//*/
//PyArrayObject* pyvector(PyObject* objin)
//{
//	return (PyArrayObject*) PyArray_ContiguousFromObject(objin,
//        NPY_DOUBLE, 1,1);
//}
///* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
//    return 1 if an error and raise exception */
//int not_doublevector(PyArrayObject* vec)
//{
//	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1) {
//		PyErr_SetString(PyExc_ValueError,
//			"In not_doublevector: array must be of type Float and 1 dimensional (n).");
//		return 1;
//	}
//	return 0;
//}
//
///* #### Matrix Extensions ############################## */
//
///* ==== Row x 2 function - manipulate matrix in place ======================
//    Multiply the 2nd row of the input by 2 and put in output
//    interface:  rowx2(mat1, mat2)
//                mat1 and mat2 are NumPy matrices
//                Returns integer 1 if successful                        */
//static PyObject* rowx2(PyObject* self, PyObject* args)
//{
//	PyArrayObject* matin, *matout;  // The python objects to be extracted from the args
//	double** cin, **cout;           // The C matrices to be created to point to the 
//	//   python matrices, cin and cout point to the rows
//	//   of matin and matout, respectively
//	int i, j, n, m;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &matin,
//		&PyArray_Type, &matout))  return NULL;
//	if (NULL == matin)  return NULL;
//	if (NULL == matout)  return NULL;
//
//	/* Check that objects are 'double' type and matrices
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublematrix(matin)) return NULL;
//	if (not_doublematrix(matout)) return NULL;
//
//	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
//	cin=pymatrix_to_Carrayptrs(matin);
//	cout=pymatrix_to_Carrayptrs(matout);
//
//	/* Get matrix dimensions. */
//	n=matin->dimensions[0];
//	m=matin->dimensions[1];
//
//	/* Operate on the matrices  */
//	for (i=0; i<n; i++) {
//		for (j=0; j<m; j++) {
//			if (i==1) cout[i][j]=2.0*cin[i][j];
//		}
//	}
//
//	/* Free memory, close file and return */
//	free_Carrayptrs(cin);
//	free_Carrayptrs(cout);
//	return Py_BuildValue("i", 1);
//}
///* ==== Row x 2 function- Version 2. - manipulate matrix in place ======================
//    Multiply the 2nd row of the input by 2 and put in output
//    interface:  rowx2(mat1, mat2)
//                mat1 and mat2 are NumPy matrices
//                Returns integer 1 if successful
//    Uses the utility function pymatrix to make NumPy C objects from PyObjects
//*/
//static PyObject* rowx2_v2(PyObject* self, PyObject* args)
//{
//	PyObject* Pymatin, *Pymatout;   // The python objects to be extracted from the args
//	PyArrayObject* matin, *matout;  // The python array objects to be extracted from python objects
//	double** cin, **cout;           // The C matrices to be created to point to the 
//	//   python matrices, cin and cout point to the rows
//	//   of matin and matout, respectively
//	int i, j, n, m;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "OO", &Pymatin, &Pymatout))  return NULL;
//	if (NULL == Pymatin)  return NULL;
//	if (NULL == Pymatout)  return NULL;
//
//	/* Convert Python Objects to Python Array Objects */
//	matin= pymatrix(Pymatin);
//	matout= pymatrix(Pymatout);
//
//	/* Check that objects are 'double' type and matrices
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublematrix(matin)) return NULL;
//	if (not_doublematrix(matout)) return NULL;
//
//	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
//	cin=pymatrix_to_Carrayptrs(matin);
//	cout=pymatrix_to_Carrayptrs(matout);
//
//	/* Get matrix dimensions. */
//	n=matin->dimensions[0];
//	m=matin->dimensions[1];
//
//	/* Operate on the matrices  */
//	for (i=0; i<n; i++) {
//		for (j=0; j<m; j++) {
//			if (i==1) cout[i][j]=2.0*cin[i][j];
//		}
//	}
//
//	/* Free memory, close file and return */
//	free_Carrayptrs(cin);
//	free_Carrayptrs(cout);
//	return Py_BuildValue("i", 1);
//}
///* ==== Square matrix components function & multiply by int and float =========
//    Returns a NEW NumPy array
//    interface:  matsq(mat1, i1, d1)
//                mat1 is NumPy matrix, i1 is Python integer, d1 is Python float (double)
//                returns a NumPy matrix                                        */
//static PyObject* matsq(PyObject* self, PyObject* args)
//{
//	PyArrayObject* matin, *matout;
//	double** cin, **cout, dfactor;
//	int i, j, n, m, dims[2], ifactor;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!id",
//		&PyArray_Type, &matin, &ifactor, &dfactor))  return NULL;
//	if (NULL == matin)  return NULL;
//
//	/* Check that object input is 'double' type and a matrix
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublematrix(matin)) return NULL;
//
//	/* Get the dimensions of the input */
//	n=dims[0]=matin->dimensions[0];
//	m=dims[1]=matin->dimensions[1];
//
//	/* Make a new double matrix of same dims */
//	matout=(PyArrayObject*) PyArray_FromDims(2, dims, NPY_DOUBLE);
//
//	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
//	cin=pymatrix_to_Carrayptrs(matin);
//	cout=pymatrix_to_Carrayptrs(matout);
//
//	/* Do the calculation. */
//	for (i=0; i<n; i++) {
//		for (j=0; j<m; j++) {
//			cout[i][j]= ifactor*dfactor*cin[i][j]*cin[i][j];
//		}
//	}
//
//	/* Free memory, close file and return */
//	free_Carrayptrs(cin);
//	free_Carrayptrs(cout);
//	return PyArray_Return(matout);
//}
//
///* ==== Operate on Matrix components as contiguous memory =========================
//  Shows how to access the array data as a contiguous block of memory. Used, for example,
//  in matrix classes implemented as contiquous memory rather than as n arrays of 
//  pointers to the data "rows"
//  
//    Returns a NEW NumPy array
//    interface:  contigmat(mat1, x1)
//                mat1 is NumPy matrix, x1 is Python float (double)
//                returns a NumPy matrix                                        */
//static PyObject* contigmat(PyObject* self, PyObject* args)
//{
//	PyArrayObject* matin, *matout;
//	double* cin, *cout, x1;     // Pointers to the contiguous data in the matrices to
//	// be used by C (e.g. passed to a program that uses
//	// matrix classes implemented as contiquous memory rather
//	// than as n arrays of pointers to the data "rows"
//	int i, j, n, m, dims[2], ncomps;  // ncomps=n*m=total number of matrix components in mat1
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!d",
//		&PyArray_Type, &matin, &x1))  return NULL;
//	if (NULL == matin)  return NULL;
//
//	/* Check that object input is 'double' type and a matrix
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_doublematrix(matin)) return NULL;
//
//	/* Get the dimensions of the input */
//	n=dims[0]=matin->dimensions[0];
//	m=dims[1]=matin->dimensions[1];
//	ncomps=n*m;
//
//	/* Make a new double matrix of same dims */
//	matout=(PyArrayObject*) PyArray_FromDims(2, dims, NPY_DOUBLE);
//
//	/* Change contiguous arrays into C * arrays pointers to PyArrayObject data */
//	cin=pyvector_to_Carrayptrs(matin);
//	cout=pyvector_to_Carrayptrs(matout);
//
//	/* Do the calculation. */
//	printf("In contigmat, cout (as contiguous memory) =\n");
//	for (i=0; i<ncomps; i++) {
//		cout[i]= cin[i]-x1;
//		printf("%e ", cout[i]);
//	}
//	printf("\n");
//
//	return PyArray_Return(matout);
//}
//
///* #### Matrix Utility functions ######################### */
//
///* ==== Make a Python Array Obj. from a PyObject, ================
//     generates a double matrix w/ contiguous memory which may be a new allocation if
//     the original was not a double type or contiguous 
//  !! Must DECREF the object returned from this routine unless it is returned to the
//     caller of this routines caller using return PyArray_Return(obj) or
//     PyArray_BuildValue with the "N" construct   !!!
//*/
//PyArrayObject* pymatrix(PyObject* objin)
//{
//	return (PyArrayObject*) PyArray_ContiguousFromObject(objin,
//        NPY_DOUBLE, 2,2);
//}
///* ==== Create Carray from PyArray ======================
//    Assumes PyArray is contiguous in memory.
//    Memory is allocated!                                    */
//double** pymatrix_to_Carrayptrs(PyArrayObject* arrayin)
//{
//	double** c, *a;
//	int i, n, m;
//
//	n=arrayin->dimensions[0];
//	m=arrayin->dimensions[1];
//	c=ptrvector(n);
//	a=(double*) arrayin->data;  /* pointer to arrayin data as double */
//	for (i=0; i<n; i++) {
//		c[i]=a+i*m;
//	}
//	return c;
//}
///* ==== Allocate a double *vector (vec of pointers) ======================
//    Memory is Allocated!  See void free_Carray(double ** )                  */
//double** ptrvector(long n)
//{
//	double** v;
//	v=(double**)malloc((size_t) (n*sizeof(double)));
//	if (!v) {
//		printf(
//			"In **ptrvector. Allocation of memory for double array failed.");
//		exit(0);
//	}
//	return v;
//}
///* ==== Free a double *vector (vec of pointers) ========================== */
//void free_Carrayptrs(double** v)
//{
//	free((char*) v);
//}
///* #### Integer 2D Array Extensions ############################## */
//
///* ==== Integer function - manipulate integer 2D array in place ======================
//    Replace >=0 integer with 1 and < 0 integer with 0 and put in output
//    interface:  intfcn1(int1, afloat)
//                int1 is a NumPy integer 2D array, afloat is a Python float
//                Returns integer 1 if successful                        */
//static PyObject* intfcn1(PyObject* self, PyObject* args)
//{
//	PyArrayObject* intin, *intout;  // The python objects to be extracted from the args
//	int** cin, **cout;              // The C integer 2D arrays to be created to point to the 
//	//   python integer 2D arrays, cin and cout point to the rows
//	//   of intin and intout, respectively
//	int i, j, n, m, dims[2];
//	double afloat;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!d",
//		&PyArray_Type, &intin, &afloat))  return NULL;
//	if (NULL == intin)  return NULL;
//
//	printf("In intfcn1, the input Python float = %e, a C double\n", afloat);
//
//	/* Check that object input is int type and a 2D array
//	Not needed if python wrapper function checks before call to this routine */
//	if (not_int2Darray(intin)) return NULL;
//
//	/* Get the dimensions of the input */
//	n=dims[0]=intin->dimensions[0];
//	m=dims[1]=intin->dimensions[1];
//
//	/* Make a new int array of same dims */
//	intout=(PyArrayObject*) PyArray_FromDims(2, dims, NPY_LONG);
//
//	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
//	cin=pyint2Darray_to_Carrayptrs(intin);
//	cout=pyint2Darray_to_Carrayptrs(intout);
//
//	/* Do the calculation. */
//	for (i=0; i<n; i++) {
//		for (j=0; j<m; j++) {
//			if (cin[i][j] >= 0) {
//				cout[i][j]= 1;
//			} else {
//				cout[i][j]= 0;
//			}
//		}
//	}
//
//	printf("In intfcn1, the output array is,\n\n");
//
//	for (i=0; i<n; i++) {
//		for (j=0; j<m; j++) {
//			printf("%d ", cout[i][j]);
//		}
//		printf("\n");
//	}
//	printf("\n");
//
//	/* Free memory, close file and return */
//	free_Cint2Darrayptrs(cin);
//	free_Cint2Darrayptrs(cout);
//	return PyArray_Return(intout);
//}
///* #### Integer Array Utility functions ######################### */
//
///* ==== Make a Python int Array Obj. from a PyObject, ================
//     generates a 2D integer array w/ contiguous memory which may be a new allocation if
//     the original was not an integer type or contiguous 
//  !! Must DECREF the object returned from this routine unless it is returned to the
//     caller of this routines caller using return PyArray_Return(obj) or
//     PyArray_BuildValue with the "N" construct   !!!
//*/
//PyArrayObject* pyint2Darray(PyObject* objin)
//{
//	return (PyArrayObject*) PyArray_ContiguousFromObject(objin,
//        NPY_LONG, 2,2);
//}
///* ==== Create integer 2D Carray from PyArray ======================
//    Assumes PyArray is contiguous in memory.
//    Memory is allocated!                                    */
//int** pyint2Darray_to_Carrayptrs(PyArrayObject* arrayin)
//{
//	int** c, *a;
//	int i, n, m;
//
//	n=arrayin->dimensions[0];
//	m=arrayin->dimensions[1];
//	c=ptrintvector(n);
//	a=(int*) arrayin->data;  /* pointer to arrayin data as int */
//	for (i=0; i<n; i++) {
//		c[i]=a+i*m;
//	}
//	return c;
//}
///* ==== Allocate a a *int (vec of pointers) ======================
//    Memory is Allocated!  See void free_Carray(int ** )                  */
//int** ptrintvector(long n)
//{
//	int** v;
//	v=(int**)malloc((size_t) (n*sizeof(int)));
//	if (!v) {
//		printf(
//			"In **ptrintvector. Allocation of memory for int array failed.");
//		exit(0);
//	}
//	return v;
//}
///* ==== Free an int *vector (vec of pointers) ========================== */
//void free_Cint2Darrayptrs(int** v)
//{
//	free((char*) v);
//}
///* ==== Check that PyArrayObject is an int (integer) type and a 2D array ==============
//    return 1 if an error and raise exception
//    Note:  Use NY_LONG for NumPy integer array, not NP_INT      */
//int not_int2Darray(PyArrayObject* mat)
//{
//	if (mat->descr->type_num != NPY_LONG || mat->nd != 2) {
//		PyErr_SetString(PyExc_ValueError,
//			"In not_int2Darray: array must be of type int and 2 dimensional (n x m).");
//		return 1;
//	}
//	return 0;
//}







// EOF
