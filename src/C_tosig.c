
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

#include "C_tosig.h"
#include <math.h>
#include "ToSig.h"
#include "_recombine.h"

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_tosigMethods[] = {
		{"stream2logsig", tologsig, METH_VARARGS,"stream2logsig(array(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, \"the data in stream space\" and returns a numpy vector containing the logsignature of the vector series up to given signature_degree"},
		{"stream2sig", tosig, METH_VARARGS, "stream2logsig(array(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, \"the data in stream space\" and returns a numpy vector containing the signature of the vector series up to given signature_degree"},
		{"logsigdim", getlogsigsize, METH_VARARGS,"logsigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the dimension of the log signature vector returned by array2logsig"},
		{"sigdim", getsigsize, METH_VARARGS,"sigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the length of the signature vector returned by array2logsig"},
		{"logsigkeys",showlogsigkeys, METH_VARARGS, "logsigkeys(signal_dimension, signature_degree) returns, in the order used by ...2logsig, a space separated ascii string containing the keys associated the entries in the log signature returned by ...2logsig"},
		{"sigkeys",showsigkeys, METH_VARARGS, "sigkeys(signal_dimension, signature_degree) returns, in the order used by ...2sig, a space separated ascii string containing the keys associated the entries in the signature returned by ...2sig"},
		{"recombine", (PyCFunction) pyrecombine, METH_VARARGS | METH_KEYWORDS, "recombine(ensemble, selector=(0,1,2,...no_points-1), weights = (1,1,..,1), degree = 1) ensemble is a numpy array of vectors of type NP_DOUBLE referred to as points, the selector is a list of indexes to rows in the ensemble, weights is a list of positive weights of equal length to the selector and defines an emirical measure on the points in the ensemble. Returns (retained_indexes, new weights) The arrays index_array, weights_array are single index numpy arrays and must have the same dimension and represent the indexes of the vectors and a mass distribution of positive weights (and at least one must be strictly positive) on them. The returned weights are strictly positive, have the same total mass - but are supported on a subset of the initial chosen set of locations. If degree has its default value of 1 then the vector data has the same integral under both weight distributions; if degree is k then both sets of weights will have the same moments for all degrees at most k; the indexes returned are a subset of indexes in input index_array and mass cannot be further recombined onto a proper subset while preserving the integral and moments. The default is to index of all the points, the default weights are 1. on each point indexed. The default degree is one."},
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

/* ==== Reduces the support of a probabiity measure on vectors to the minimal support size with the same expectation/ moments <= degree=========================
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
		size_t* LOCS = PyArray_DATA(src_locations);
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
		double* WTS = PyArray_DATA(src_weights);
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
	double* DATA = PyArray_DATA(data);

	size_t* LOCATIONS = PyArray_DATA(src_locations);
	double* WEIGHTS = PyArray_DATA(src_weights);

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
	ptrdiff_t noKeptLocations = NoDimensionsToCubature;
	// a buffer of size iNoDimensionsToCubature big enough to store array of indexes to the kept points
	size_t* KeptLocations = (size_t*)malloc(noKeptLocations * sizeof(size_t));
	// a buffer of size NoDimensionsToCubature to store the weights of the kept points
	double* NewWeights = (double*)malloc(noKeptLocations * sizeof(double));

	_recombineC(
		stCubatureDegree
		, point_dimension
		, no_locations
		, &noKeptLocations
		, LOCATIONS2
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
	PyArrayObject* snk_locations = (PyArrayObject*)PyArray_SimpleNew(1, d, NPY_INTP);
	PyArrayObject* snk_weights = (PyArrayObject*)PyArray_SimpleNew(1, d, NPY_DOUBLE);
	// MOVE OUTPUT FROM BUFFERS TO THESE OBJECTS
	memcpy(PyArray_DATA(snk_locations), KeptLocations, noKeptLocations * sizeof(size_t));
	memcpy(PyArray_DATA(snk_weights), NewWeights, noKeptLocations * sizeof(double));
	// RELEASE BUFFERS
	free(KeptLocations);
	free(NewWeights);
	// CREATE OUTPUT
	out = PyTuple_Pack(2, snk_locations, snk_weights);
exit:;
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
