/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




#ifndef DJC_COROPA_LIBALGEBRA_POLYLIEH_SEEN
#define DJC_COROPA_LIBALGEBRA_POLYLIEH_SEEN



/// The Lie algebra for the commutative polynomials.

/// Elements of the algebra are polynomial vector fields (ie linear combinations
/// of pairs of monomials and directions). The product is the Lie bracket
/// for vector fields.

template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class poly_lie : public algebra<poly_lie_basis<SCA, RAT, n_letters, max_degree> >
{
public:
	/// The basis elements for the polynomials (monomials)
	typedef typename poly_basis<SCA, RAT>::KEY POLYBASISKEY;
	/// The basis type.
	typedef poly_lie_basis<SCA, RAT, n_letters, max_degree> BASIS;
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
	/// Default constructor. Zero lie element.
	poly_lie(void) {}
	/// Copy constructor.
	poly_lie(const poly_lie& l)
		: ALG(l) {}
	/// Constructs an instance from an algebra instance.
	poly_lie(const ALG& a)
		: ALG(a) {}
	/// Constructs an instance from a sparse_vector instance.
	poly_lie(const VECT& v)
		: ALG(v) {}
	/// Constructs a unidimensional instance from a given key (with scalar one).
	explicit poly_lie(LET x, LET y, DEG z)
		: ALG ()
	{
		POLYBASISKEY tempkey;
		tempkey[ y ] = z;
		ALG tempalg(KEY (x, tempkey));
		ALG:: swap(tempalg);
	}
	/// Constructs an instance from a basis element.
	explicit poly_lie(const KEY& k)
		: ALG(k) {}
	/// Constructs a unidimensional instance from a letter and a scalar.
	explicit poly_lie(LET letter, const SCA& s)
		: ALG(VECT::basis.keyofletter(letter), s) {}
public:
	/// Replaces the occurences of letters in s by Lie elements in v.
	inline friend poly_lie replace(const poly_lie& src, const std::vector<LET>& s, const std::vector<poly_lie*>& v)	
	{
		poly_lie result;
		std::map<KEY, poly_lie> table;
		const_iterator i;
		for (i = src.begin(); i != src.end(); ++i)
			result.add_scal_prod(VECT::basis.replace(i->first, s, v, table), i->second);
		return result;
	}
};
#endif
