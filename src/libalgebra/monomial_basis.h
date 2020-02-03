/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  monomial_basis.h


// Include once wrapper
#ifndef monomial_basisH_SEEN
#define monomial_basisH_SEEN

/// Implements an interface for the set of words of a finite number of letters.
/** 

    A basis is a finite total ordered set of keys, its cardinal is size() and
	its minimal element is begin(). The successor key of a given key is given
	by nextkey(). The successor of the maximal key is end() and does not
	belong to the basis. The position of a given key in the total order of the
	basis is given by keypos(), and equals 1 for begin(). To each letter
	corresponds a key.

	The monomial_basis is a basis for which keys are words of letters, totally
	ordered by the lexicographical order. Letters corresponds to words of
	length one. The empty word is a special key which serves for the embeding
	of scalars into the tensor algebra. The begin() key is the empty word. 

	Implementation: a key is implemeted using dual ended queue of letters.
	Letters are natural numbers, enumerated starting from 1, in such a way
	that n_letters is the biggest letter. We use the lexicographical order.
	The invalid key made with one occurrence of letter 0 is used for end().

    There is no key product here since the monomial_basis class serves as a
	common ancestor for monomial_basis and shuffle_tensor_basis classes.
	The implementation of the prod() member function constitutes the essential
	difference between monomial_basis and shuffle_tensor_basis.
*/
template<typename SCA, DEG n_letters, DEG max_degree>
class monomial_basis
{
public:
	/// A key is an STD dual ended queue of letters.
	typedef std::deque<LET> KEY;
	/// A default key corresponds to the empty word.
	const KEY empty_key;
	/// The MAP type.
	typedef std::map<KEY, SCA> MAP;
private:
	/// The size
	DEG _size;
public:
	/// Default constructor. Empty basis.
	monomial_basis(void)
	{
		// We statically compute the size (n^{d+1}-1)/(n-1) 
		// where n == n_letters and d == max_degree.
		_size = 1;
		for (DEG i = 1; i <= max_degree; ++i)
			_size += _size * n_letters;	
	}
public:
	/// Returns the key corresponding to a letter.
	inline KEY keyofletter(LET letter) const
	{
		KEY result;
		result.push_back(letter);
		return result;
	}
	/// Tells if a key is a letter (i.e. word of length one). 
	inline bool letter(const KEY& k) const
	{
		return k.size() == 1;
	}
	/// Returns the first letter of a key.
	inline LET getletter(const KEY& k) const
	{
		return k[0];
	}
	/// Returns the first letter of a key. For compatibility with lie_basis.
	inline KEY lparent(const KEY& k) const
	{
		KEY result;
		result.push_back(k[0]);
		return result;
	}
	/// Returns the key which corresponds to the sub-word after the first letter.
	inline KEY rparent(const KEY& k) const
	{
		KEY result(k);
		result.pop_front();
		return result;
	}
	/// Returns the length of the key viewed as a word of letters.
	inline DEG degree(const KEY& k) const
	{
		return k.size();
	}
	/// Returns the size of the basis.
	inline DEG size(void) const
	{
		return _size;
	}
	/// Computes the position of a key in the basis total order (it has a cost).
	inline DEG keypos(const KEY& k) const
	{
		DEG pos = 1;
		for (DEG i = 0; i < k.size(); ++i)
			pos += n_letters * (k[i] - 1);
		return pos;
	}
	/// Returns the value of the smallest key in the basis.
	inline KEY begin(void) const
	{
		return empty_key;
	}
	/// Returns the key next the biggest key of the basis.
	inline KEY end(void) const
	{
		KEY result; // empty key.
		result.push_back(0); // invalid key.	
		return result;
	}
	/// Returns the key next a given key in the basis.
	inline KEY nextkey(const KEY& k) const
	{
		KEY::size_type i;
		//TJL: error - i never gets negative
		for (i = k.size() - 1; i /*>= 0*/ < k.size(); --i)
			if (k[i] < n_letters)
			{
				KEY result(k);
				result[i] += 1;
				return result;
			}
		return end();
	}
	/// Outputs a key as a string of letters to an std::ostringstream.
	std::string key2string(const KEY& k) const
	{
		std::ostringstream oss;
		KEY::size_type i;
		if (!k.empty())
			oss << k[0];
		for (i = 1; i < k.size(); ++i)
			oss << "," << k[i];
		return oss.str();
	}
};

/// The monoid of words of a finite number of letters with concat product.
/** 
	This is the basis used to implement the multi_polynomial class as a
	specialisation of the algebra class. This basis is the Free Associative
	Algebra basis with a finite number of letters, with the usual
	concatenation product. The monomial_basis is a container of keys. A key
	is the implementation of a word of letters. The prod() member function
	corresponds to the concatenation of the two keys given as arguments. This
	product is associative but not commutative. Letters can be seen as
	particular basis keys, i.e. words of length one. The empty word is a
	special key used for the imbedding of letters (words of length one).
*/
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class free_monomial_basis : public monomial_basis<SCA, n_letters, max_degree>
{
public:
	/// The monomial_basis type.
	typedef monomial_basis<SCA, n_letters, max_degree> PBASIS;
	/// Import of the KEY type.
	typedef typename PBASIS::KEY KEY;
	/// Import of the MAP type.
	typedef typename PBASIS::MAP MAP;
	/// The rationals.
	typedef RAT RATIONAL;
	/// The multi_polynomial element type.
	typedef multi_polynomial<SCA, RAT, n_letters, max_degree> MULTIPOLY;
public:
	/// Default constructor.
	free_monomial_basis(void) {}
public:
	/// The concatenation product of two basis elements.
	/**
	Returns the multi_polynomial obtained by the concatenation product of two keys
	viewed as words of letters. The result is a unidimensional multi_polynomial
	with a unique key (the concatenation of k1 and k2) associated to the +1
	scalar. The already computed products are not stored or remembered.
	*/
	inline MULTIPOLY prod(const KEY& k1, const KEY& k2) const
	{
		static SCA one(+1);
		MULTIPOLY result;
		if ((max_degree == 0) || (k1.size() + k2.size() <= max_degree))
		{
			KEY concat(k1);
			for (typename KEY::size_type i = 0; i < k2.size(); ++i)
				concat.push_back(k2[i]);
			result[concat] = one;
		}
		return result;
	}
	/// Outputs an std::pair<free_monomial_basis*, KEY> to an std::ostream.
	inline friend
		std::ostream& operator<<(std::ostream& os, const std::pair<free_monomial_basis*, KEY>& t)
	{
		return os << (t.first)->key2string(t.second);
	}
};

// Include once wrapper
#endif

//EOF.
