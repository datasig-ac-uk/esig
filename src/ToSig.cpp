// ToSig.cpp : Defines the exported functions for the DLL application.
//
#include "ToSig.h" //Python.h must come first
#include "stdafx.h"

#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <libalgebra/libalgebra.h>
#include <libalgebra/coefficients/coefficients.h>
#include "libalgebra/lie_basis.h"

//#include <lie_basis.h>
namespace {

	typedef double S;
	typedef double Q;

    using field_t = alg::coefficients::double_field;

    template <unsigned Width, unsigned Depth>
    using lie_type = alg::lie<field_t, Width, Depth, alg::vectors::dense_vector>;

    template <unsigned Width, unsigned Depth>
    using tensor_type = alg::free_tensor<field_t, Width, Depth, alg::vectors::dense_vector>;

    template <unsigned Width, unsigned Depth>
    using cbh_type = alg::cbh<field_t, Width, Depth, tensor_type<Width, Depth>, lie_type<Width, Depth>>;

    template <unsigned Width, unsigned Depth>
    using maps_type = alg::maps<field_t, Width, Depth, tensor_type<Width, Depth>, lie_type<Width, Depth>>;

    template <unsigned Width, unsigned Depth>
    using lie_data_access = alg::vectors::dtl::data_access < alg::vectors::dense_vector < alg::lie_basis<Width, Depth>, field_t>>;

    template <unsigned Width, unsigned Depth>
    using tensor_data_access = alg::vectors::dtl::data_access<alg::vectors::dense_vector<alg::free_tensor_basis<Width, Depth>, field_t>>;


  /**
   * row_to_lie - replaces vector_to_lie
   * @param stream pointer to stream as PyArrayObject, assumed to have two dimensions, the row is assumed to be of length WIDTH
   * @param rowId index of row to be converted to LIE
   * @return row as LIE (the entries in the row are coefficients of the letters) 
   */
	template <unsigned Width, unsigned Depth>
	lie_type<Width, Depth> row_to_lie(PyArrayObject *stream, npy_intp rowId)
	{
//        auto *begin = reinterpret_cast<const double *>(PyArray_GETPTR2(stream, rowId, 0));
//        auto *end = reinterpret_cast<const double *>(PyArray_GETPTR2(stream, rowId + 1, 0));
//        lie_type<Width, Depth> ans(begin, end);
        lie_type<Width, Depth> ans;
		for (alg::LET i = 1; i <= static_cast<alg::LET>(Width); ++i) {
            ans.add_scal_prod(i, *reinterpret_cast<const S *>(PyArray_GETPTR2(stream, rowId, static_cast<npy_intp>(i - 1))));
        }
		return ans;
	}
  
 /*
	template <class LIE, class STATE, size_t WIDTH>
	LIE vector_to_lie(const STATE& arg)
	{
		LIE ans;
		for (alg::LET i = 1; i <= WIDTH; ++i)
			ans += LIE(i, arg[i - 1]);
		return ans;
	}
  */

  /**
   * GetLogSignature
   * @param stream pointer to stream as PyArrayObject, assumed to have two dimensions, the row is assumed to be of length WIDTH
   * @return the log signature of the stream as LIE
   */
	template <unsigned Width, unsigned Depth>
	lie_type<Width, Depth> GetLogSignature(PyArrayObject *stream)
    {
//        auto* real_stream = reinterpret_cast<PyArrayObject*>(
//            PyArray_FromAny(
//                reinterpret_cast<PyObject*>(stream),
//                PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
//                NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
//                nullptr)
//            );
        npy_intp numRows = PyArray_DIM(stream, 0);

        try {
            std::vector<lie_type<Width, Depth>> increments;
            increments.reserve(numRows-1);
            if (numRows > 0) {
                npy_intp rowId = 0;
                auto previous = row_to_lie<Width, Depth>(stream, rowId++);

                for (; rowId < numRows; ++rowId) {
                    auto next(row_to_lie<Width, Depth>(stream, rowId));
                    increments.push_back(next - previous);
                    previous = next;
                }
            }
            cbh_type<Width, Depth> cbh;
            return (!increments.empty()) ? cbh.full(increments.begin(), increments.end()) : lie_type<Width, Depth>();
        } catch (...) {
//            Py_DECREF(real_stream);
            throw;
        }
	}
  /*
	template <class LIE, class STATE, class CBH, size_t WIDTH>
	LIE GetLogSignature(const STATE* begin, const STATE* end)
	{
		std::vector<LIE> increments;
		if (begin != end) {
			const STATE* it = begin;
			LIE previous = vector_to_lie<LIE, STATE, WIDTH>(*(it++));

			for (; it != end; ++it) {
				LIE next(vector_to_lie<LIE, STATE, WIDTH> (*it));
				increments.push_back(next - previous);
				previous = next;
			}
		}
		std::vector<LIE*> pincrements;
		for (typename std::vector<LIE>::iterator it = increments.begin();
			it != increments.end(); ++it)
			pincrements.push_back(&(*it));
		CBH cbh;
		return (pincrements.size() != 0) ? cbh.full(pincrements) : LIE();
	}
  */

  /**
   * KeyToIndexRec - recursive helper function used in KeyToIndex
   */
	template <class TENSOR, size_t WIDTH>
	inline std::pair<size_t, typename TENSOR::BASIS::KEY> KeyToIndexRec(size_t i,typename TENSOR::BASIS::KEY k){	
		return k.size() ? KeyToIndexRec<TENSOR,WIDTH>(i*WIDTH+k.FirstLetter(),k.rparent()) : std::make_pair(i,k) ;
	}

  /**
   * KeyToIndex - computes the position of a key in the vectorised tensor
   * @param k tensor key
   * @return position as index
   */
	template <class TENSOR, size_t WIDTH>
	inline size_t KeyToIndex(typename TENSOR::BASIS::KEY k)
	{		
		return KeyToIndexRec<TENSOR,WIDTH>(0,k).first;
		//return (k.size()) ? k.FirstLetter() + WIDTH * KeyToIndex<TENSOR, WIDTH>(k.rparent()) : 0; // incorrect version, commented out
	}
  /*
	template <class TENSOR, size_t WIDTH>
	inline size_t KeyToIndex(typename TENSOR::BASIS::KEY k)
	{
		return (k.size()) ? k.FirstLetter() + WIDTH * KeyToIndex<TENSOR,
			WIDTH>(k.rparent()) : 0;
	}
  */

	template <class VECTOR, class TENSOR, size_t WIDTH>
	struct fn0001 {
		VECTOR& _ans;
		fn0001(VECTOR& ans):_ans(ans)
		{
		}
#ifndef LIBALGEBRA_VECTORS_H
		template <class T>
        void operator()(T& element)
        {
            _ans[KeyToIndex<TENSOR, WIDTH>(element.first)] = element.second;
        }
#else
        template <typename T>
        void operator()(T& element)
        {
		    _ans[KeyToIndex<TENSOR, WIDTH>(element.key())] = element.value();
        }
#endif

	};

	//[&ans] (const decltype(*(arg.begin()))& element){
	//	ans[KeyToIndex<TENSOR, WIDTH>(element.first)] = element.second;
	//}


	template <unsigned Width, unsigned Depth>
	void unpack_tensor_to_SNK(const tensor_type<Width, Depth>& arg, PyArrayObject *snk)
	{
        const auto& base = alg::vectors::dtl::vector_base_access::convert(arg);
        auto* range_begin = tensor_data_access<Width, Depth>::range_begin(base);
        auto* range_end = tensor_data_access<Width, Depth>::range_end(base);

        std::copy(range_begin, range_end, reinterpret_cast<double*>(PyArray_DATA(snk)));
	}

	template <class VECTOR>
	struct fn0002 {
		VECTOR& _ans;
		fn0002(VECTOR& ans):_ans(ans)
		{
		}

#ifndef LIBALGEBRA_VECTORS_H
		template <class T>
        void operator()(T& element)
        {
            _ans[element.first - 1] = element.second;
        }
#else
        template <class T>
        void operator()(T& element)
        {
            _ans[element.key() - 1] = element.value();
        }
#endif
	};

	template <unsigned Width, unsigned Depth>
	void unpack_lie_to_SNK(const lie_type<Width, Depth>& arg, PyArrayObject *snk)
	{
        const auto& base = alg::vectors::dtl::vector_base_access::convert(arg);
        auto* range_begin = lie_data_access<Width, Depth>::range_begin(base);
        auto* range_end = lie_data_access<Width, Depth>::range_end(base);

        std::copy(range_begin, range_end, reinterpret_cast<double*>(PyArray_DATA(snk)));
	}

	template <unsigned Width, unsigned Depth>
	std::string liebasis2stringT()
	{
		//typedef alg::free_tensor<S, Q, WIDTH, DEPTH> TENSOR;
        using LIE = lie_type<Width, Depth>;

		std::string ans;
		for (typename LIE::BASIS::KEY k = LIE::basis.begin(); k != LIE::basis.end();
			k = LIE::basis.nextkey(k))
			ans += std::string(" ") + LIE::basis.key2string(k);
		return ans;
	}

	template <unsigned Width, unsigned Depth>
	std::string tensorbasis2stringT()
	{
        //typedef alg::lie<S, Q, WIDTH, DEPTH> LIE;
        using TENSOR = tensor_type<Width, Depth>;

        std::string ans;
		for (typename TENSOR::BASIS::KEY k = TENSOR::basis.begin();
			k < TENSOR::basis.end(); k = TENSOR::basis.nextkey(k))
			ans += std::string(" (") + TENSOR::basis.key2string(k) +
				std::string(")");
		return ans;
	}

  /**
   * GetSigT - computes the signature of a stream into snk
   * @param stream pointer to stream as PyArrayObject, assumed to have two dimensions, the row is assumed to be of length WIDTH
   * @param snk pointer to C array, the result is written into this array
   */
	template <unsigned Width, unsigned Depth>
	bool GetSigT(PyArrayObject *stream, PyArrayObject *snk)
	{
		auto logans = GetLogSignature<Width, Depth>(stream);
		maps_type<Width, Depth> maps;
		auto signature = exp(maps.l2t(logans));
		unpack_tensor_to_SNK(signature, snk);
		return true;
	}

  /*
	template <size_t WIDTH, size_t DEPTH>
	bool GetSigT(const double* src, double* snk, size_t recs)
	{
		typedef const double array[WIDTH];
		const array* source = reinterpret_cast<const array*>(src);
		typedef double S;
		typedef double Q;
		typedef alg::free_tensor<S, Q, WIDTH, DEPTH> TENSOR;
		typedef alg::lie<S, Q, WIDTH, DEPTH> LIE;
		typedef alg::maps<S, Q, WIDTH, DEPTH> MAPS;
		typedef alg::cbh<S, Q, WIDTH, DEPTH> CBH;
		LIE logans = GetLogSignature<LIE, array, CBH, WIDTH>(&source[0],
			&source[0 + recs]);
		MAPS maps;
		TENSOR signature = exp(maps.l2t(logans));
		unpack_tensor_to_SNK<S, TENSOR, WIDTH, DEPTH>(signature, snk);
		return true;
	}
  */

  /**
   * GetSigT - computes size of the vectorised signature
   * @return size of the vectorised signature
   */
	template <size_t WIDTH, size_t DEPTH>
	size_t GetSigT()
	{
		const size_t unpacked_tensor_dimension = (WIDTH > 1)
			? ((size_t(alg::ConstPower < WIDTH, DEPTH + 1 > ::ans) - 1) / (WIDTH -
			1))
			: WIDTH * DEPTH + 1 ;
		return unpacked_tensor_dimension;
	}

  /**
   * GetLogSigT - computes size of the vectorised log-signature
   * @return size of the vectorised log-signature
   */
	template <unsigned Width, unsigned Depth>
	size_t GetLogSigT()
	{
        return lie_type<Width, Depth>::BASIS::start_of_degree(Depth+1);
	}

  /**
   * GetLogSigT - computes the log-signature of a stream into snk
   * @param stream pointer to stream as PyArrayObject, assumed to have two dimensions, the row is assumed to be of length WIDTH
   * @param snk pointer to C array, the result is written into this array
   */
	template <unsigned Width, unsigned Depth>
	bool GetLogSigT(PyArrayObject *stream, PyArrayObject *snk)
	{
		auto logans = GetLogSignature<Width, Depth>(stream);
		unpack_lie_to_SNK(logans, snk);
		return true;
	}

  /*
	template <size_t WIDTH, size_t DEPTH>
	bool GetLogSigT(const double* src, double* snk, size_t recs)
	{
		typedef const double array[WIDTH];
		const array* source = reinterpret_cast<const array*>(src);
		typedef double S;
		typedef double Q;
		typedef alg::lie<S, Q, WIDTH, DEPTH> LIE;
		typedef alg::cbh<S, Q, WIDTH, DEPTH> CBH;
		LIE logans = GetLogSignature<LIE, array, CBH, WIDTH>(&source[0],
			&source[0 + recs]);
		unpack_lie_to_SNK<S, LIE, WIDTH, DEPTH>(logans, snk);
		return true;
	}
*/
}

// A C++ function returning a string of labels
extern TOSIG_API std::string ShowLogSigLabels(size_t width, size_t depth)
{
	//execute the correct Templated Function and return the value
	try {
#define TemplatedFn(depth,width) liebasis2stringT<depth,width>()
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
	// only get here if the template arguments are out of range
	return std::string();
}

// A C++ function returning a string of labels
extern TOSIG_API std::string ShowSigLabels(size_t width, size_t depth)
{
	//execute the correct Templated Function and return the value
	try {
#define TemplatedFn(depth,width) tensorbasis2stringT<depth,width>()
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
	// only get here if the template arguments are out of range
	return std::string();
}


// compute log signature of path at src and place answer in snk
TOSIG_API int GetLogSig(PyArrayObject *stream, PyArrayObject *snk,
    size_t width, size_t depth)
 {
    try {
    //execute the correct Templated Function and return the value
#define TemplatedFn(depth,width) GetLogSigT<depth,width>(stream, snk)
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
    // only get here if the template arguments are out of range
    return false;
 }

// get required size for snk
TOSIG_API size_t GetLogSigSize(size_t width, size_t depth)
 {
    //execute the correct Templated Function and return the value
    try {
#define TemplatedFn(depth,width) GetLogSigT<depth,width>()
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
    // only get here if the template arguments are out of range
    return 0;
 }

// a wrapper un-templated function that calls the correct template instance
TOSIG_API int GetSig(PyArrayObject *stream, PyArrayObject *snk,
    size_t width, size_t depth)
 {
    //execute the correct Templated Function and return the value
    try {
#define TemplatedFn(depth,width) GetSigT<depth,width>(stream, snk)
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
    // only get here if the template arguments are out of range
    return false;
 }

// get required size for snk
TOSIG_API size_t GetSigSize(size_t width, size_t depth)
 {
    //execute the correct Templated Function and return the value
    try {
#define TemplatedFn(depth,width) GetSigT<depth,width>()
#include "switch.h"
#undef TemplatedFn
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
    // only get here if the template arguments are out of range
    return 0;
 }
