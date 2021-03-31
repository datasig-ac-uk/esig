//
// Created by sam on 31/03/2021.
//

/* A file to test importing C modules for handling arrays to Python */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
//https://github.com/numpy/numpy/issues/9309
// there are big problems in defining these
// python/numpy functions in different translation units
// so we wont!

#include <stddef.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include "ToSig.h"

#ifndef ESIG_NO_RECOMBINE
#include "_recombine.h"
#endif

extern "C" {
/* Header to test of C modules for arrays for Python: C_tosig.c */

// API: http://docs.scipy.org/doc/numpy/reference/c-api.array.html
// and http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#c.NpyIter_New

/* ==== Prototypes ================================== */
/* .... The ToSig Functionality ......................*/
static PyObject *tologsig(PyObject *self, PyObject *args);
static PyObject *tosig(PyObject *self, PyObject *args);
static PyObject *getlogsigsize(PyObject *self, PyObject *args);
static PyObject *getsigsize(PyObject *self, PyObject *args);
#ifndef ESIG_NO_RECOMBINE
static PyObject *pyrecombine(PyObject *self, PyObject *args, PyObject *keywds);
#endif
PyObject *showsigkeys(PyObject *self, PyObject *args);
PyObject *showlogsigkeys(PyObject *self, PyObject *args);


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

};


/* #### Globals #################################### */

PyDoc_STRVAR(stream2logsig_doc,
"stream2logsig(array(no_of_ticks x signal_dimension),"
" signature_degree) reads a 2 dimensional numpy array"
" of floats, \"the data in stream space\" and returns"
" a numpy vector containing the log signature of the"
" vector series up to given log signature degree"
);

PyDoc_STRVAR(stream2sig_doc,
"stream2logsig(array(no_of_ticks x signal_dimension),"
" signature_degree) reads a 2 dimensional numpy array"
" of floats, \"the data in stream space\" and returns"
" a numpy vector containing the signature of the vector"
" series up to given signature degree"
);

PyDoc_STRVAR(logsigdim_doc,
"logsigdim(signal_dimension, signature_degree) returns"
" a Py_ssize_t integer giving the dimension of the log"
" signature vector returned by stream2logsig"
);

PyDoc_STRVAR(sigdim_doc,
"sigdim(signal_dimension, signature_degree) returns"
" a Py_ssize_t integer giving the length of the"
" signature vector returned by stream2logsig"
);

PyDoc_STRVAR(logsigkeys_doc,
"logsigkeys(signal_dimension, signature_degree) returns,"
" in the order used by stream2logsig, a space separated ascii"
" string containing the keys associated the entries in the"
" log signature returned by stream2logsig"
);

PyDoc_STRVAR(sigkeys_doc,
"sigkeys(signal_dimension, signature_degree) returns,"
" in the order used by stream2sig, a space separated ascii"
" string containing the keys associated the entries in"
" the signature returned by stream2sig"
);
#ifndef ESIG_NO_RECOMBINE
PyDoc_STRVAR(recombine_doc,
"recombine(ensemble, selector=(0,1,2,...no_points-1),"
" weights = (1,1,..,1), degree = 1) ensemble is a numpy"
" array of vectors of type NP_DOUBLE referred to as"
" points, the selector is a list of indexes to rows in"
" the ensemble, weights is a list of positive weights of"
" equal length to the selector and defines an empirical"
" measure on the points in the ensemble."
" Returns (retained_indexes, new weights) The arrays"
" index_array, weights_array are single index numpy arrays"
" and must have the same dimension and represent the indexes"
" of the vectors and a mass distribution of positive weights"
" (and at least one must be strictly positive) on them."
" The returned weights are strictly positive, have the"
" same total mass - but are supported on a subset of the"
" initial chosen set of locations. If degree has its default"
" value of 1 then the vector data has the same integral under"
" both weight distributions; if degree is k then both sets of"
" weights will have the same moments for all degrees at most k;"
" the indexes returned are a subset of indexes in input"
" index_array and mass cannot be further recombined onto a"
" proper subset while preserving the integral and moments."
" The default is to index of all the points, the default"
" weights are 1. on each point indexed."
" The default degree is one."
);
#endif


/* ==== Set up the methods table ====================== */
static PyMethodDef _C_tosigMethods[] = {
        {"stream2logsig", tologsig, METH_VARARGS, stream2logsig_doc},
        {"stream2sig", tosig, METH_VARARGS, stream2sig_doc},
        {"logsigdim", getlogsigsize, METH_VARARGS, logsigdim_doc},
        {"sigdim", getsigsize, METH_VARARGS, sigdim_doc},
        {"logsigkeys",showlogsigkeys, METH_VARARGS, logsigkeys_doc},
        {"sigkeys",showsigkeys, METH_VARARGS, sigkeys_doc},
#ifndef ESIG_NO_RECOMBINE
        {"recombine", (PyCFunction) pyrecombine, METH_VARARGS | METH_KEYWORDS, recombine_doc},
#endif
        {NULL, NULL, 0, NULL}        /* Sentinel */
};


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

// Module init function for Python 3.x only.
PyMODINIT_FUNC
PyInit_tosig(void){

    PyObject *m;

    m = PyModule_Create(&moduledef);

    // Needed for using numpy arrays in the module
    import_array();




    return m;
}





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
    // SM 8/10/20: added error handling to the switch statement
    // to be handled here
    if (!GetLogSig(seriesin, vecout, width, depth))
        return NULL;

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
    // SM 8/10/20: added error handling to the switch statement
    // to be handled here
    if (!GetSig(seriesin, vecout, width, depth))
        return NULL;

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

    // SM 8/10/20: added error handling to the switch statement
    // to be handled here
    if (ans == 0)
        return NULL;

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

    // SM 8/10/20: added error handling to the switch statement
    // to be handled here
    if (ans == 0)
        return NULL;

    return Py_BuildValue("n", ans);
}

#ifndef ESIG_NO_RECOMBINE
/* ==== Reduces the support of a probability measure on vectors to the minimal support size with the same
 * expectation/ moments <= degree=========================
	Returns two the new probability measure via two NEW scalar NumPy arrays of same length indices (Py_ssize_t) and weights (double)
	interface:  py_recombine(N_vectors_of_dimension_D(N,D) and optionally: , k_indices, k_weights, degree = 1)
				indices are Py_INTP; vectors and weights are doubles, degree is int
				returns n_retained_indices, n_new_weights. If degree > 1 then the memory requirement (D^degree * N + D^(degree*2) ) and complexity (D^degree * N + D^(degree*3) * log(N/D^degree) grow VERY rapidly .
				*/
static PyObject*
pyrecombine(PyObject* self, PyObject* args, PyObject* keywds)
{
// INTERNAL
    //
    int src_locations_built_internally = 0;
    int src_weights_built_internally = 0;
    // match the mean - or higher moments
    size_t stCubatureDegree;
    // max no points at end - computed below
    ptrdiff_t NoDimensionsToCubature;
    // parameter summaries
    ptrdiff_t no_locations;
    ptrdiff_t point_dimension;
    double total_mass = 0.;
    // the final output
    PyObject* out = NULL;

// THE INPUTS
    // the data - a (large) enumeration of vectors obtained by making a list of vectors and converting it to an array.
    PyArrayObject* data;
    // a list of the rows of interest
    PyArrayObject* src_locations = NULL;
    // their associated weights
    PyArrayObject* src_weights = NULL;
    // match the mean - or higher moments
    ptrdiff_t CubatureDegree = 1;

    // Pre declaration of variables that are only used in the main branch.
    // The compiler is complaining when variables are declared and initialised
    // between the goto and label exit
    PyArrayObject *snk_locations = NULL;
    PyArrayObject *snk_weights = NULL;
    double* NewWeights;
    size_t* LOCATIONS;
    ptrdiff_t noKeptLocations;
    size_t* KeptLocations;

    // usage def recombine(array1, *args, degree=1)
    static char* kwlist[] = { "ensemble", "selector", "weights", "degree" , NULL };
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|O!O!n:recombine", kwlist, &PyArray_Type, &data, &PyArray_Type, &src_locations, &PyArray_Type, &src_weights, &CubatureDegree))
        return out;
// DATA VALIDATION
    //
    if (data == NULL
        || (PyArray_NDIM(data) != 2 || PyArray_DIM(data, 0) == 0 || PyArray_DIM(data, 1) == 0) // present but badly formed
        || (src_locations != NULL && (PyArray_NDIM(src_locations) != 1 || PyArray_DIM(src_locations, 0) == 0)) // present but badly formed
        || (src_weights != NULL && (PyArray_NDIM(src_weights) != 1 || PyArray_DIM(src_weights, 0) == 0)) // present but badly formed
        ||((src_weights != NULL && src_locations != NULL) && !PyArray_SAMESHAPE(src_weights, src_locations) )// present well formed but of different length
        || CubatureDegree < 1
            ) return NULL;
    stCubatureDegree = CubatureDegree; //(convert from signed to unsigned)
// create default locations (ALL) if not specified
    if (src_locations == NULL) {
        npy_intp* d = PyArray_DIMS(data);
        //d[0] = PyArray_DIM(data, 0);
        src_locations = (PyArrayObject*)PyArray_SimpleNew(1, d, NPY_INTP);
        size_t* LOCS = reinterpret_cast<size_t*>(PyArray_DATA(src_locations));
        ptrdiff_t id;
        for (id = 0; id < d[0]; ++id)
            LOCS[id] = id;
        src_locations_built_internally = 1;
    }
// create default weights (1. on each point) if not specified
    if (src_weights == NULL) {
        npy_intp d[1];
        d[0] = PyArray_DIM(src_locations, 0);
        src_weights = (PyArrayObject*)PyArray_SimpleNew(1, d, NPY_DOUBLE);
        double* WTS = reinterpret_cast<double*>(PyArray_DATA(src_weights));
        ptrdiff_t id;
        for (id = 0; id < d[0]; ++id)
            WTS[id] = 1.;
        src_weights_built_internally = 1;
    }
// make all data contiguous and type compliant (This only applies to external data - we know that our created arrays are fine
// note this requires a deref at the end - so does the fill in of defaults - but we only do one or the other
    {
        data = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)data, NPY_DOUBLE, 2, 2);
        if (!src_locations_built_internally)
            src_locations = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)src_locations, NPY_INTP, 1, 1);
        if (!src_weights_built_internally)
            src_weights = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)src_weights, NPY_DOUBLE, 1, 1);
    }


// PREPARE INPUTS AS C ARRAYS
    ptrdiff_t no_datapoints = PyArray_DIM(data, 0);
    point_dimension = PyArray_DIM(data, 1);
    double* DATA = reinterpret_cast<double*>(PyArray_DATA(data));

    LOCATIONS = reinterpret_cast<size_t*>(PyArray_DATA(src_locations));
    double* WEIGHTS = reinterpret_cast<double*>(PyArray_DATA(src_weights));

// map locations from integer indexes to pointers to double
    no_locations = PyArray_DIM(src_locations, 0);
    double** LOCATIONS2 = (double**)malloc(no_locations * sizeof(double*));
    ptrdiff_t id;
    for (id = 0; id < no_locations; ++id)
    {
        // check for data out of range
        if (LOCATIONS[id] >= no_datapoints)
            goto exit;
        LOCATIONS2[id] = &DATA[LOCATIONS[id] * point_dimension];
    }
    // normalize the weights
    for (id = 0; id < no_locations; ++id)
        total_mass += WEIGHTS[id];
    for (id = 0; id < no_locations; ++id)
        WEIGHTS[id]/= total_mass;


// NoDimensionsToCubature = the max number of points needed for cubature
    _recombineC(
            stCubatureDegree
            , point_dimension
            , 0 // tells _recombineC to return NoDimensionsToCubature the required buffer size
            , &NoDimensionsToCubature
            , NULL
            , NULL
            , NULL
            , NULL
    );
// Prepare to call the reduction algorithm
    // a variable that will eventually be amended to to indicate the actual number of points returned
    noKeptLocations = NoDimensionsToCubature;
    // a buffer of size iNoDimensionsToCubature big enough to store array of indexes to the kept points
    KeptLocations = (size_t*)malloc(noKeptLocations * sizeof(size_t));
    // a buffer of size NoDimensionsToCubature to store the weights of the kept points
    NewWeights = (double*)malloc(noKeptLocations * sizeof(double));

    _recombineC(
            stCubatureDegree
            , point_dimension
            , no_locations
            , &noKeptLocations
            , (const void**) LOCATIONS2
            , WEIGHTS
            , KeptLocations
            , NewWeights
    );
    // un-normalise the weights
    for (id = 0; id < noKeptLocations; ++id)
        NewWeights[id] *= total_mass;
// MOVE ANSWERS TO OUT
    // MAKE NEW OUTPUT OBJECTS
    npy_intp d[1];
    d[0] = noKeptLocations;

    snk_locations = (PyArrayObject *) PyArray_SimpleNew(1, d, NPY_INTP);
    snk_weights = (PyArrayObject *) PyArray_SimpleNew(1, d, NPY_DOUBLE);
    // MOVE OUTPUT FROM BUFFERS TO THESE OBJECTS
    memcpy(PyArray_DATA(snk_locations), KeptLocations, noKeptLocations * sizeof(size_t));
    memcpy(PyArray_DATA(snk_weights), NewWeights, noKeptLocations * sizeof(double));
    // RELEASE BUFFERS
    free(KeptLocations);
    free(NewWeights);
    // CREATE OUTPUT
    out = PyTuple_Pack(2, snk_locations, snk_weights);


    exit:
// CLEANUP
    free(LOCATIONS2);
    Py_DECREF(data);
    Py_DECREF(src_locations);
    Py_DECREF(src_weights);
// EXIT
    return out;
// USEFUL NUMPY EXAMPLES
    //https://stackoverflow.com/questions/56182259/how-does-one-acces-numpy-multidimensionnal-array-in-c-extensions/56233469#56233469
    //https://stackoverflow.com/questions/6994725/reversing-axis-in-numpy-array-using-c-api/6997311#6997311
    //https://stackoverflow.com/questions/6994725/reversing-axis-in-numpy-array-using-c-api/6997311#699731
}

#endif


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

