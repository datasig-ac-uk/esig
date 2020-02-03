/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  poly_basis.h


// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_POLYBASISH_SEEN
#define DJC_COROPA_LIBALGEBRA_POLYBASISH_SEEN

/// A polynomial basis class.
/**
This is the basis used to implement the polynomial class as a
specialisation of the algebra class. A key implements a monomial in several
variables, with scalar coefficient one. Each variable corresponds to a
letter. The product of monomial, provided by the member function prod(), is
the standard commutative product of monomials. The type LET is used to
number the variables (i.e. letters). The type SCA is used to evaluate the
monomial. In the current implementation, no key is stored in memory, only
the member functions are provided.

An empty monomial corresponds to a constant term, i.e. a scalar SCA.
*/
template<typename SCA, typename RAT>
class poly_basis : public basis_traits<With_Degree>
{
public:
	/// A key is a map from letters to degrees (i.e. a monomial of letters).
	typedef std::map<LET, DEG> KEY;
	/// A default key corresponds to the monomial of degree 0.
	const KEY empty_key;
	/// The rationals.
	typedef RAT RATIONAL;
	struct KEY_LESS;
	/// The MAP type.
	typedef std::map<KEY, SCA, KEY_LESS> MAP;
	/// The Multivariate Polynomials Algebra element type.
	typedef poly<SCA, RAT> POLY;
public:
	/// Default constructor. Empty basis.
	poly_basis(void) {}
public:
	// tjl the code below is strange - 
	// the result of such an evaluation should be a scalar times a key of unevaluated variables
	/// Evaluate the key from a vector of scalars.
	inline SCA eval_key(const KEY& k, const std::map<LET, SCA>& values) const
	{
		SCA result(1);
		typename KEY::const_iterator it;
		for (it = k.begin(); it != k.end(); ++it)
			if (it->second > 0)
			{
				typename std::map<LET, SCA>::const_iterator iit;
				iit = values.find(it->first);
				try
				{	
					if (iit != values.end())
					{
						for (DEG j = 1; j <= it->second; ++j)
							result *= iit->second;	
					}
					else
					{
						throw "not all variables have values!";
					}

				}
				catch (char* str)
				{
					std::cerr << "Exception raised: " << str << '\n';
					abort();
				}

			}
			

		return result;
	}
	///Multiplication of two monomials, outputted as a monomial
	inline static KEY prod2(const KEY& k1, const KEY& k2)
	{
		KEY k(k1);
		typename KEY::const_iterator it;
		for (it = k2.begin(); it != k2.end(); ++it)
		{
			k[it->first] += it->second;
		}
		return k;
	}
	/// Returns the polynomial corresponding to the product of two keys (monomials).
	/**
	For polynomials, this product is unidimensional, i.e. it 
	is a key since the product of two monomials (keys) is a monomial 
	(key) again. To satisfy the condtions of algebra, the output is 
	in the form of a polynomial.
	*/
	inline static POLY prod(const KEY& k1, const KEY& k2)
	{
		POLY result;
		result[prod2(k1, k2)] = ((SCA)+1);
		return result;
	}
	/// Returns the key (monomial) corresponding to a letter (variable).
	inline KEY keyofletter(LET letter) const
	{
		KEY result;
		result[letter] = +1;
		return result;
	}
	/// Returns the degree of a monomial
	inline static DEG degree(const KEY& k)
	{
		DEG result (0);
		typename KEY::const_iterator it;
		for (it = k.begin(); it != k.end(); ++it)
		{
			result += DEG(it->second);
		}
		return result;
	}
	
	struct  KEY_LESS
	{
		bool inline operator()(const KEY& lhs, const KEY& rhs) const
		{
			return ((degree(lhs) < degree(rhs)) || ((degree(lhs) == degree(rhs)) && lhs < rhs));
		}
	};	

	
	/// Returns the value of the smallest key in the basis.
	inline KEY begin(void) const
	{
		return empty_key;
	}
	///// Returns the key next the biggest key of the basis. 
	//// this doesn't make sense without a maximum degree.
	//inline KEY end(void) const
	//{
	//KEY result; // empty key.
	//result.push_back(0); // invalid key.
	//return result;
	//}
	///// Returns the key next a given key in the basis.
	//// We need an ordering for this to work
	//inline KEY nextkey(const KEY& k) const
	//{
	//KEY::size_type i;
	//for (i = k.size()-1; i >= 0; --i)
	//if (k[i]<n_letters) { KEY result(k); result[i] += 1; return result; }
	//return end();
	//}
	
	/// Outputs a std::pair<poly_basis*, KEY> to an std::ostream.
	inline friend
		std::ostream& operator<<(std::ostream& os, const std::pair<poly_basis*, KEY>& t)
	{
		bool first(true);
		typename KEY::const_iterator it;
		for (it = t.second.begin(); it != t.second.end(); ++it)
			if (it->second > 0)
			{
				if (!first)
					os << ' ';
				else
					first = false;
				os << "x" << it->first;
				if (it->second > 1)
					os << '^' << it->second;
			}
		return os;
	}
};

// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_POLYBASISH_SEEN

//EOF.
