/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//multi_polynomial.h


#ifndef multi_polynomialH_SEEN
#define multi_polynomialH_SEEN

/// A specialisation of the algebra class with a free tensor basis.
/**
   Mathematically, the algebra of multi_polynomial instances is a free associative
   algebra. With respect to the inherited algebra class, the essential
   distinguishing feature of this class is the basis class used, and in
   particular the basis::prod() member function. Thus, the most important
   information is in the definition of monomial_basis. Notice that this
   associative algebra of free tensors includes as a sub-algebra the
   associative algebra corresponding to the SCALAR type. This is permitted by
   the existence of empty keys in monomial_basis.
 */
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class multi_polynomial : public algebra<free_monomial_basis<SCA, RAT, n_letters, max_degree> >
{
public:
	/// The basis type.
	typedef free_monomial_basis<SCA, RAT, n_letters, max_degree> BASIS;
	/// Import of the KEY type.
	typedef typename BASIS::KEY KEY;
	/// The sparse_vector type.
	typedef sparse_vector<BASIS> VECT;
	/// The algebra type.
	typedef algebra<BASIS> ALG;
	/// Import of the iterator type.
	typedef typename ALG::iterator iterator;
	/// Import of the constant iterator type.
	typedef typename ALG::const_iterator const_iterator;
public:
	/// Default constructor.
	multi_polynomial(void) {}
	/// Copy constructor.
	multi_polynomial(const multi_polynomial& t)
		: ALG(t) {}
	/// Constructs an instance from an algebra instance.
	multi_polynomial(const ALG& a)
		: ALG(a) {}
	/// Constructs an instance from a sparse_vector instance.
	multi_polynomial(const VECT& v)
		: ALG(v) {}	
	/// Constructs a unidimensional instance from a letter and a scalar.
	multi_polynomial(LET letter, const SCA& s)
		: ALG(VECT::basis.keyofletter(letter), s) {}
	/// Explicit unidimensional constructor from a given key (basis element).
	explicit multi_polynomial(const KEY& k)
		: ALG(k) {}
	/// Explicit unidimensional constructor from a given scalar.
	explicit multi_polynomial(const SCA& s)
		: ALG(VECT::basis.empty_key, s) {}
public:
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_BINARY_OPERATOR(multi_polynomial,*,*=,SCA)
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_BINARY_OPERATOR(multi_polynomial,/,/=,RAT)
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_BINARY_OPERATOR(multi_polynomial,*,*=,multi_polynomial)
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_BINARY_OPERATOR(multi_polynomial,+,+=,multi_polynomial)
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_BINARY_OPERATOR(multi_polynomial,-,-=,multi_polynomial)
	/// Ensures that the return type is a multi_polynomial.
  inline __DECLARE_UNARY_OPERATOR(multi_polynomial,-,-,ALG)
	/// Computes the truncated exponential of a multi_polynomial instance.
	inline friend multi_polynomial exp(const multi_polynomial& arg)
	{
		// Computes the truncated exponential of arg
		// 1 + arg + arg^2/2! + ... + arg^n/n! where n = max_degree
		static KEY kunit;
		multi_polynomial result(kunit);
		for (DEG i = max_degree; i >= 1; --i)
		{
			result.mul_scal_div(arg, (RAT)i);
			result += (multi_polynomial)kunit;
		}
		return result;
	}
	/// Computes the truncated logarithm of a multi_polynomial instance.
	inline friend multi_polynomial log(const multi_polynomial& arg)
	{
		// Computes the truncated log of arg up to degree max_degree
		// The coef. of the constant term (empty word in the monoid) of arg 
		// is forced to 1.
		// log(arg) = log(1+x) = x - x^2/2 + ... + (-1)^(n+1) x^n/n.
		// max_degree must be > 0
		static KEY kunit;
		multi_polynomial tunit(kunit);
		multi_polynomial x(arg);
		iterator it = x.find(kunit);
		if (it != x.end())
			x.erase(it);
		multi_polynomial result;
		for (DEG i = max_degree; i >= 1; --i)
		{
			if (i % 2 == 0)
				result.sub_scal_div(tunit, (RAT)i);
			else
				result.add_scal_div(tunit, (RAT)i);
			result *= x;
		}
		return result;
	}
};

// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_TENSORH_SEEN

//EOF.
