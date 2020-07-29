#ifndef ToSig_h__
#define ToSig_h__
#include <Python.h> 
#include "stdlib.h" //size_t
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

// define TOSIG_LINKAGE to DYNAMIC if one is using a dll 
#ifdef TOSIG_LINKAGE
#if TOSIG_LINKAGE == DYNAMIC
#define TOSIG_API DLLTOSIG_API
#endif
#else
#define TOSIG_API
#endif

#if defined (__GNUC__) && defined(__unix__)
#define DLLTOSIG_API __attribute__ ((__visibility__("default")))
#elif defined (WIN32)
#ifdef DLLTOSIG_EXPORTS
#define DLLTOSIG_API __declspec(dllexport)
#else
#define DLLTOSIG_API __declspec(dllimport)
#endif
#else
#define DLLTOSIG_API
#endif


#ifdef __cplusplus
#include <string>
extern TOSIG_API std::string ShowLogSigLabels(size_t width, size_t depth);
extern TOSIG_API std::string ShowSigLabels(size_t width, size_t depth);
extern "C" {
#endif

	// get required size for snk
	const TOSIG_API size_t GetSigSize(size_t width, size_t depth);
	// compute signature of path at src and place answer in snk
	int TOSIG_API GetSig(PyArrayObject *stream, PyArrayObject *snk, 
		size_t width, size_t depth);

	// get required size for snk
	size_t TOSIG_API GetLogSigSize(size_t width, size_t depth);
	// compute signature of path at src and place answer in snk
	int TOSIG_API GetLogSig(PyArrayObject *stream, PyArrayObject *snk, 
		size_t width, size_t depth);
#ifdef __cplusplus
}
#endif

#endif // ToSig_h__
