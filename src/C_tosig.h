#ifndef C_tosig_h__
#define C_tosig_h__

#ifdef __cplusplus
extern "C" {
#endif
/* Header to test of C modules for arrays for Python: C_tosig.c */

// API: http://docs.scipy.org/doc/numpy/reference/c-api.array.html
// and http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#c.NpyIter_New

/* ==== Prototypes ================================== */
/* .... The ToSig Functionality ......................*/
static PyObject* tologsig(PyObject* self, PyObject* args);
static PyObject* tosig(PyObject* self, PyObject* args);
static PyObject* getlogsigsize(PyObject* self, PyObject* args);
static PyObject* getsigsize(PyObject* self, PyObject* args);
static PyObject* pyrecombine(PyObject* self, PyObject* args, PyObject* keywds);
PyObject * showsigkeys(PyObject *self, PyObject *args);
PyObject * showlogsigkeys(PyObject *self, PyObject *args);

/* .... Python callable Vector functions ............* /
static PyObject *vecfcn1(PyObject *self, PyObject *args);
static PyObject *vecsq(PyObject *self, PyObject *args);*/

/* .... C vector utility functions ..................*/
PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int  not_doublevector(PyArrayObject *vec);


/* .... Python callable Matrix functions ..................* /
static PyObject *rowx2(PyObject *self, PyObject *args);
static PyObject *rowx2_v2(PyObject *self, PyObject *args);
static PyObject *matsq(PyObject *self, PyObject *args);
static PyObject *contigmat(PyObject *self, PyObject *args);*/

/* .... C matrix utility functions ..................*/
PyArrayObject *pymatrix(PyObject *objin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);
int  not_doublematrix(PyArrayObject *mat);

/* .... Python callable integer 2D array functions ..................* /
static PyObject *intfcn1(PyObject *self, PyObject *args);*/

/* .... C 2D int array utility functions ..................*/
PyArrayObject *pyint2Darray(PyObject *objin);
int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin);
int **ptrintvector(long n);
void free_Cint2Darrayptrs(int **v);
int  not_int2Darray(PyArrayObject *mat);
#ifdef __cplusplus
}
#endif
#endif // C_tosig_h__
