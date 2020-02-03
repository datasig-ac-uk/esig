/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */

#ifndef alg_types_h__
#define alg_types_h__

#include "libalgebra/libalgebra.h"
#include "addons/gmpwrapper.h"
//#include "mtl/mtl.h"

//#pragma warning(push)
//#pragma warning (disable : 800)
//#include "../addons/gmpwrapper.h"
//#pragma warning(pop)


enum coefficient_t
{
	Rational,
	DPReal,
	SPReal
};

namespace {

	template <coefficient_t F> struct Field;

	template<>
	struct Field < Rational >
	{
		typedef mpq_class S;
		typedef mpq_class Q;
	};

	template<>
	struct Field < DPReal >
	{
		typedef double S;
		typedef double Q;
	};

	template<>
	struct Field < SPReal >
	{
		typedef float S;
		typedef float Q;
	};

} // anon namespace

template <size_t D, size_t W, coefficient_t F = Rational> 
struct alg_types : Field < F >
{
	typedef typename Field < F >::S S;
	typedef typename Field < F >::Q Q;
	typedef S SCA;
	typedef Q RAT;
	typedef alg::DEG DEG;
	typedef alg::LET LET;
	static const unsigned DEPTH = D;
	static const unsigned myDIM = W;
	static const unsigned ALPHABET_SIZE = W;
	typedef alg::poly<S,Q> MULTIPOLY1;
	typedef alg::free_tensor<S,Q,ALPHABET_SIZE,DEPTH> TENSOR;
	typedef alg::lie<S,Q,ALPHABET_SIZE,DEPTH> LIE;
	typedef alg::maps<S,Q,ALPHABET_SIZE,DEPTH> MAPS;
	typedef alg::cbh<S,Q,ALPHABET_SIZE,DEPTH> CBH;
	typedef alg::multi_polynomial<S,Q,ALPHABET_SIZE,DEPTH> MULTIPOLY;
	//typedef mtl::dense1D<RAT> mtlVector;
	//typedef typename mtl::matrix<RAT, mtl::rectangle<>, mtl::dense<>, mtl::row_major>::type mtlMatrix;
	//typedef typename mtl::matrix<RAT, mtl::diagonal<>, mtl::packed<>, mtl::row_major>::type mtlDiagMat;
};



////  alg_types.h : provides an interface to and sets consistent sets of basic algebraic types
//
//
//#ifndef alg_types_h__
//#define alg_types_h__
//
//#include "libalgebra.h"
//
//#pragma warning(push)
//#pragma warning (disable : 800)
//#include "../addons/gmpwrapper.h"
//#pragma warning(pop)
//
//
//enum coefficient_t
//{
//	Rational,
//	DPReal,
//	SPReal
//};
//
//namespace
//{
//
//template <coefficient_t F>
//struct Field;
//
//template<>
//struct Field<Rational>
//{
//	typedef mpq_class S;
//	typedef mpq_class Q;
//};
//
//template<>
//struct Field<DPReal>
//{
//	typedef double S;
//	typedef double Q;
//};
//
//template<>
//struct Field<SPReal>
//{
//	typedef float S;
//	typedef float Q;
//};
//
//} // anon namespace
//
//template <unsigned D, unsigned W, coefficient_t F = Rational>
//struct alg_types : Field<F>
//{
//	typedef typename Field<F>::S S;
//	typedef typename Field<F>::Q Q;
//	typedef S SCA;
//	typedef Q RAT;
//	static const unsigned DEPTH = D;
//	static const unsigned myDIM = W;
//	static const unsigned ALPHABET_SIZE = W;
//	typedef alg::poly<S, Q> MULTIPOLY1;
//	typedef alg::free_tensor<S, Q, ALPHABET_SIZE, DEPTH> TENSOR;
//	typedef alg::lie<S, Q, ALPHABET_SIZE, DEPTH> LIE;
//	typedef alg::maps<S, Q, ALPHABET_SIZE, DEPTH> MAPS;
//	typedef alg::cbh<S, Q, ALPHABET_SIZE, DEPTH> CBH;
//	typedef alg::multi_polynomial<S, Q, ALPHABET_SIZE, DEPTH> MULTIPOLY;
//};

#endif // alg_types_h__
