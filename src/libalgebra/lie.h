/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  lie.h


// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_LIEH_SEEN
#define DJC_COROPA_LIBALGEBRA_LIEH_SEEN

/// A specialisation of the algebra class with a Lie basis.
/**
   Mathematically, the algebra of Lie instances is a free Lie associative
   algebra. With respect to the inherited algebra class, the essential
   distinguishing feature of this class is the basis class used, and in
   particular the basis::prod() member function. Thus, the most important
   information is in the definition of lie_basis. Notice that this associative
   algebra of lie elements does not includes as a sub-algebra the associative
   algebra corresponding to the SCALAR type. In other words, only the scalar
   zero corresponds to a Lie element (the zero one) which is the neutral
   element of the addition operation. There is no neutral element for the
   product (free Lie product).
 */
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class lie : public algebra<lie_basis<SCA, RAT, n_letters, max_degree> >
{
public:
	/// The basis type.
	typedef lie_basis<SCA, RAT, n_letters, max_degree> BASIS;
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
	lie(void) {}
	/// Copy constructor.
	lie(const lie& l)
		: ALG(l) {}
	/// Constructs an instance from an algebra instance.
	lie(const ALG& a)
		: ALG(a) {}
	/// Constructs an instance from a sparse_vector instance.
	lie(const VECT& v)
		: ALG(v) {}
	/// Constructs a unidimensional instance from a given key (with scalar one).
	explicit lie(const KEY& k)
		: ALG(k) {}
	/// Constructs a unidimensional instance from a letter and a scalar.
	explicit lie(LET letter, const SCA& s)
		: ALG(VECT::basis.keyofletter(letter), s) {}
public:
	/// Replaces the occurrences of letters in s by Lie elements in v.
	inline friend lie replace(const lie& src, const std::vector<LET>& s, const std::vector<lie*>& v)	
	{
		lie result;
		std::map<KEY, lie> table;
		const_iterator i;
		for (i = src.begin(); i != src.end(); ++i)
			result.add_scal_prod(VECT::basis.replace(i->first, s, v, table), i->second);
		return result;
	}
};

// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_LIEH_SEEN

//EOF.
