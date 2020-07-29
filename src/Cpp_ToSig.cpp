#define PY_SSIZE_T_CLEAN
#include "Python.h" // MUST BE FIRST BY STANDARD
#include "stdafx.h"//nh

#include <string>
#include <map>
#include <utility>


extern std::string ShowLogSigLabels(size_t width, size_t depth);
extern std::string ShowSigLabels(size_t width, size_t depth);

extern "C" {
	PyObject *
		showlogsigkeys(PyObject *self, PyObject *args)
	{
		typedef std::pair<size_t,size_t> SIGTYPE;
		typedef std::map<SIGTYPE, std::string> DICT;
		static DICT theLieBasesStrngs;

		Py_ssize_t depth, width;

		/* Parse tuple */
		if (!PyArg_ParseTuple(args, "nn",
			&width, &depth))  return NULL;
		SIGTYPE sigtype(width,depth); 

		DICT::const_iterator it = theLieBasesStrngs.find(sigtype);
		if (it == theLieBasesStrngs.end()) 
			return Py_BuildValue("s",(theLieBasesStrngs[sigtype] = ShowLogSigLabels(width, depth)).c_str());
		else
			return Py_BuildValue("s", (it->second).c_str());
	}

	PyObject *
		showsigkeys(PyObject *self, PyObject *args)
	{
		typedef std::pair<size_t,size_t> SIGTYPE;
		typedef std::map<SIGTYPE, std::string> DICT;
		static DICT theTensorBasesStrngs;

		Py_ssize_t depth, width;

		/* Parse tuple */
		if (!PyArg_ParseTuple(args, "nn",
			&width, &depth))  return NULL;
		SIGTYPE sigtype(width,depth); 

		DICT::const_iterator it = theTensorBasesStrngs.find(sigtype);
		if (it == theTensorBasesStrngs.end()) 
			return Py_BuildValue("s",(theTensorBasesStrngs[sigtype] = ShowSigLabels(width, depth)).c_str());
		else
			return Py_BuildValue("s", (it->second).c_str());
	}

}