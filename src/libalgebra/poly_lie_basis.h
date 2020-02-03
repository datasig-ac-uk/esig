/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




#ifndef DJC_COROPA_LIBALGEBRA_POLYLIEBASISH_SEEN
#define DJC_COROPA_LIBALGEBRA_POLYLIEBASISH_SEEN
#include "basis_traits.h"

/// A basis for the polynomial Lie algebra, poly_lie.

/** A basis for the polynomial lie algebra. 
The basis elements are vector fields of the of the form std::pair(direction, monomial).
The product is the Lie bracket of two vector fields.
*/


template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class poly_lie_basis : public basis_traits<With_Degree, n_letters, max_degree>
{
public:
	/// The basis elements of poly_basis.
	typedef poly_basis<SCA, RAT> POLYBASIS;
	/// The key of poly_basis (ie a monomial).
	typedef typename POLYBASIS::KEY POLYBASIS_KEY;
	/// A key is a pair of letter and monomial (ie a monomial in direction letter).
	typedef std::pair<LET, POLYBASIS_KEY> KEY;
	/// Polynomial algebra.
	typedef poly<SCA, RAT> POLY;
	/// The rationals.
	typedef RAT RATIONAL;
	/// The order in the MAP class reflects the degree
	struct  KEY_LESS
	{
		bool inline operator()(const KEY& lhs, const KEY& rhs) const
		{
			return ((degree(lhs) < degree(rhs)) || ((degree(lhs) == degree(rhs)) && lhs < rhs));
		}
	};	
	/// The MAP type.
	typedef std::map<KEY, SCA, KEY_LESS> MAP;
	/// The Multivariate Polynomials Algebra element type.
	typedef poly_lie<SCA, RAT, n_letters, max_degree> POLY_LIE;

public:
	/// Default constructor. Empty basis.
	poly_lie_basis(void) {}
public:

	///Returns the Lie bracket of two monomial vector fields.

	/**
	Returns the Vector field corresponding to the Lie bracket of two vector fields.
	If we have have monomials m1 and m2 and Letters i and j then the product
	of m1*d/dxi with m2*d/dxj is equal to
	m1*(d/dxi m2)*d/dxj - m2*(d/dxj m1)*d/dxi
	*/

	inline static POLY_LIE prod(const KEY& k1, const KEY& k2)
	{
		POLY poly1 = POLY::prediff(k2.second, k1.first);
		POLY poly2 = POLY::prediff(k1.second, k2.first);
		KEY mon1 (k2.first, k1.second);
		KEY mon2 (k1.first, k2.second);
		POLY_LIE result;
		result = prod2(poly1, mon1) - prod2(poly2, mon2);
		return result;
	}

	/// Multiplication of a polynomial poly1 by a monomial vector field liemon1.
	inline static POLY_LIE prod2(const POLY& poly1, const KEY& liemon1) 	
	{
		POLY_LIE result;
		for (typename POLY::const_iterator it = poly1.begin(); it != poly1.end(); it++)
		{
			SCA temp = it->second;
			POLYBASIS_KEY temp2 = POLYBASIS::prod2(liemon1.second, it->first);
			result[make_pair(liemon1.first, temp2)] = temp;
		}
		return result;
	}


	/// Turns a d/dx_i into a polynomial vector field by multiplying the empty monomial by d/dx_i.
	inline static KEY keyofletter(LET letter)
	{
		POLYBASIS empty;
		KEY result (letter, empty.empty_key);
		return result;
	}
	/// Returns the degree of the monomial in the pair (Let, monomial)
	inline static DEG degree(const KEY& k)
	{
		return POLYBASIS::degree(k.second);
	}
	/// Outputs a std::pair<poly_basis*, KEY> to an std::ostream.
	inline friend
		std::ostream& operator<<(std::ostream& os, const std::pair<poly_lie_basis*, KEY>& t)
	{
		POLYBASIS poly1;
		std::pair<POLYBASIS*, POLYBASIS_KEY> polypair;
		polypair.first = &poly1;
		polypair.second = t.second.second;
		os << "{" << polypair << "}" << "d/dx" << t.second.first << "}";
		return os;
	}


};
#endif
