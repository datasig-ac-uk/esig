/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  tensor_basis.h


// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_TENSORBASISH_SEEN
#define DJC_COROPA_LIBALGEBRA_TENSORBASISH_SEEN


#include "_tensor_basis.h"
#include "constpower.h"
#include "basis_traits.h"

/// Implements an interface for the set of words of a finite number of letters.
/** 

A basis is a finite total ordered set of keys, its cardinal is size() and
its minimal element is begin(). The successor key of a given key is given
by nextkey(). The successor of the maximal key is end() and does not
belong to the basis. The position of a given key in the total order of the
basis is given by keypos(), and equals 1 for begin(). To each letter
corresponds a key.

The tensor_basis is a basis for which keys are words of letters, totally
ordered by the lexicographical order. Letters corresponds to words of
length one. The empty word is a special key which serves for the embedding
of scalars into the tensor algebra. The begin() key is the empty word. 

Implementation: a key is implemeted using dual ended queue of letters.
Letters are natural numbers, enumerated starting from 1, in such a way
that n_letters is the biggest letter. We use the lexicographical order.
The invalid key made with one occurrence of letter 0 is used for end().

There is no key product here since the tensor_basis class serves as a
common ancestor for free_tensor_basis and shuffle_tensor_basis classes.
The implementation of the prod() member function constitutes the essential
difference between free_tensor_basis and shuffle_tensor_basis.
*/
template<typename SCA, DEG n_letters, DEG max_degree>
class tensor_basis
{
public:
	/// A key is an dual ended queue of letters
	typedef _tensor_basis<n_letters, max_degree> KEY;
	/// A default key corresponds to the empty word.
	const KEY empty_key;
	/// The MAP type.
#ifndef ORDEREDMAP
	typedef MY_UNORDERED_MAP <KEY, SCA, typename KEY::hash> MAP;
#else
#ifdef NOBTREE
	typedef std::map<KEY, SCA> MAP;
#else
    typedef btree::safe_btree_map<KEY, SCA, std::less<KEY>, std::allocator<std::pair<const KEY, SCA> >, 256> MAP;
#endif
#endif // !ORDEREDMAP


private:
	/// The size of the full basis
	LET _size;
public:
	/// Default constructor. Empty basis.

	/// We statically compute the size (n^{d+1}-1)/(n-1) 
	/// where n == n_letters and d == max_degree.
	tensor_basis(void)
		: _size ((LET(ConstPower < n_letters, max_degree + 1 > ::ans) - 1) / (n_letters - 1)) {}
public:
	/// Returns the key corresponding to a letter.
	inline KEY keyofletter(LET letter) const
	{
		return KEY(letter);
	}
	/// Tells if a key is a letter (i.e. word of length one). 
	inline bool letter(const KEY& k) const
	{
		return k.size() == 1;
	}
	/// Returns the first letter of a key.
	inline LET getletter(const KEY& k) const
	{
		return k.FirstLetter();
	}
	/// Returns the first letter of a key. For compatibility with lie_basis.
	inline KEY lparent(const KEY& k) const
	{
		return k.lparent();
	}
	/// Returns the key which corresponds to the sub-word after the first letter.
	inline KEY rparent(const KEY& k) const
	{
		return k.rparent();
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
	/// Returns the value of the smallest key in the basis.
	inline KEY begin(void) const
	{
		return empty_key;
	}
	/// Returns the key next the biggest key of the basis.
	inline KEY end(void) const
	{
		//TJL 21/08/2012
		//KEY result; // empty key.
		//result.push_back(0); // invalid key.	
		return KEY::end();
	}

	/// Returns the key next a given key in the basis.
	inline KEY nextkey(const KEY& k) const
	{
		//tjl  25/08/2012
		size_t i = k.size();
		KEY result(k);
		for (size_t j = 0 ; j < i ; ++j)
			{
				if ( k[i-1-j] < LET(n_letters))
				{					
					result[i-1-j] += 1;
					return result;
				}
				else 
					result[i-1-j] = KEY(LET(1)).FirstLetter();
			}
		if (k.size() == max_degree) 
			return end();
		else 
			return KEY(LET(1)) * result;
	}
	/// Outputs a key as a string of letters to an std::ostringstream.
	std::string key2string(const KEY& k) const
	{
		std::ostringstream oss;

		if (k.size() > 0)
		{
			oss << k.FirstLetter();
			KEY kk = k.rparent();
			for (unsigned i = 1; i < k.size(); ++i)
			{
				oss << "," << kk.FirstLetter();
				kk = kk.rparent();
			}
		}
		return oss.str();
	}
};

/// The monoid of words of a finite number of letters with concat product.
/** 
This is the basis used to implement the free_tensor class as a
specialisation of the algebra class. This basis is the Free Associative
Algebra basis with a finite number of letters, with the usual
concatenation product. The free_tensor_basis is a container of keys. A key
is the implementation of a word of letters. The prod() member function
corresponds to the concatenation of the two keys given as arguments. This
product is associative but not commutative. Letters can be seen as
particular basis keys, i.e. words of length one. The empty word is a
special key used for the imbedding of letters (words of length one).
*/
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class free_tensor_basis : public tensor_basis<SCA, n_letters, max_degree>,
						  public basis_traits<With_Degree, n_letters, max_degree>
{
public:
	/// The tensor_basis type.
	typedef tensor_basis<SCA, n_letters, max_degree> TBASIS;
	/// The rationals.
	typedef RAT RATIONAL;
	/// The Free Associative Algebra element type.
	typedef free_tensor<SCA, RAT, n_letters, max_degree> TENSOR;
public:
	/// Default constructor.
	free_tensor_basis(void) {}
public:
	/// The concatenation product of two basis elements.
	/**
	Returns the free_tensor obtained by the concatenation product of two keys
	viewed as words of letters. The result is a unidimensional free_tensor
	with a unique key (the concatenation of k1 and k2) associated to the +1
	scalar. The already computed products are not stored or remembered.
	*/

	//static inline TENSOR prod(const typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY& k1, 
	//	const typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY& k2)
	//{
	//	SCA one(+1);
	//	TENSOR result;
	//	if ((max_degree == 0) || (k1.size() + k2.size() <= max_degree))
	//	{
	//		//typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY concat = k1 * k2;
	//		result[k1 * k2] = one;
	//	}
	//	else
	//	{
	//		throw;
	//	}
	//	return result;
	//}

static inline typename TBASIS::KEY prod(const typename TBASIS::KEY& k1,
	const typename TBASIS::KEY& k2)
	{
	assert((max_degree == 0) || (k1.size() + k2.size() <= max_degree));
	return k1 * k2;
	}

//static inline TENSOR& prod(const typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY& k1,
//	const typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY& k2,
//	TENSOR& ans = (TENSOR()))
//{
//	assert((max_degree == 0) || (k1.size() + k2.size() <= max_degree));	
//	ans[k1 * k2] = SCA(+1);
//	return ans;
//}

	/// Outputs an std::pair<free_tensor_basis*, KEY> to an std::ostream.
	inline friend
		std::ostream& operator<<(std::ostream& os, const std::pair<free_tensor_basis*, typename alg::tensor_basis<SCA, n_letters, max_degree>::KEY>& t)
	{
		return os << (t.first)->key2string(t.second);
	}
};

///The monoid of words of a finite number of letters with shuffle product.
/** 
This is the basis used to implement the shuffle_tensor class as a
specialisation of the algebra class. This basis is the Free Associative
Algebra basis with a finite number of letters, with the shuffle product.
The shuffle_tensor_basis is a container of keys. A key is the
implementation of a word of letters. The prod() member function
corresponds to the shuffle product of the two keys given as arguments.
This product is associative and commutative. Letters can be seen as
particular basis keys, i.e. words of length one. The empty word is a
special key used for the imbedding of letters (words of length one).
*/
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class shuffle_tensor_basis : public tensor_basis<SCA, n_letters, max_degree>,
							 public basis_traits<With_Degree, n_letters, max_degree>
{
public:
	/// The tensor_basis type.
	typedef tensor_basis<SCA, n_letters, max_degree> TBASIS;
	/// Import of the KEY type.
	typedef typename TBASIS::KEY KEY;
	/// Import of the MAP type.
	typedef typename TBASIS::MAP MAP;
	/// The rationals.
	typedef RAT RATIONAL;
	/// The Shuffle Associative Algebra elements type.
	typedef shuffle_tensor<SCA, RAT, n_letters, max_degree> TENSOR;
public:
	/// Default constructor.
	shuffle_tensor_basis(void) {}
public:
	/// The shuffle product of two basis elements.
	/**
	Returns the shuffle_tensor obtained by the concatenation product of two
	keys viewed as words of letters. The result is a unidimensional
	shuffle_tensor with a unique key (the concatenation of k1 and k2)
	associated to the +1 scalar. The already computed products are stored in
	a static mutiplication table to speed up further calculations.
	*/
	inline const TENSOR& prod(const KEY& k1, const KEY& k2) const
	{		
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread 
		boost::lock_guard<boost::recursive_mutex> lock(table_access); 

		typedef std::map<std::pair<KEY, KEY>, TENSOR> TABLE_T;
		static TABLE_T table;
		typename TABLE_T::iterator it;
		std::pair<KEY, KEY> p(std::min(k1, k2), std::max(k1, k2));
		it = table.find(p);
		if (it == table.end())
			return table[p] = _prod(k1, k2);
		else
			return it->second;
	}
	/// Outputs a std::pair<shuffle_tensor_basis*, KEY> to an std::ostream.
	inline friend
		std::ostream& operator<<(std::ostream& os, const std::pair<shuffle_tensor_basis*, KEY>& t)
	{
		return os << (t.first)->key2string(t.second);
	}
private:
	
	inline void tensor_add_mul(const TENSOR& a, const TENSOR& b, TENSOR& ans) const
	{
		typename sparse_vector<free_tensor_basis<SCA, RAT, n_letters, max_degree>,MAP>
		::const_iterator i, j;
		for (i = a.begin(); i != a.end(); ++i)
			for (j = b.begin(); j != b.end(); ++j)
				ans.add_scal_prod(free_tensor_prod(i->first, j->first),
					i->second * j->second);
	}
	/// The concatenation product of two basis elements.
	/**
	Returns the free_tensor obtained by the concatenation product of two keys
	viewed as words of letters. The result is a unidimensional free_tensor
	with a unique key (the concatenation of k1 and k2) associated to the +1
	scalar. The already computed products are not stored or remembered.
	*/
	static inline TENSOR free_tensor_prod(const KEY& k1, const KEY& k2)
	{
		static SCA one(+1);
		TENSOR result;
		if ((max_degree == 0) || (k1.size() + k2.size() <= max_degree))
		{
			KEY concat = k1 * k2;
			result[concat] = one;
		}
		return result;
	}

	/// Computes recursively the shuffle product of two keys 
	inline TENSOR _prod(const KEY& k1, const KEY& k2) const
	{
		TENSOR result;
		//unsigned i, j;
		const SCA one(+1);

		if ((max_degree == 0) || (k1.size() + k2.size() <= max_degree))

		{
			if (k1.size() == 0)
			{
				result[k2] = one;
				return result;
			}
			if (k2.size() == 0)
			{
				result[k1] = one;
				return result;
			}
			// k1.size() >= 1 and k2.size() >= 1
			tensor_add_mul((TENSOR)k1.lparent(), prod(k1.rparent(), k2), result);
			tensor_add_mul((TENSOR)k2.lparent(), prod(k1, k2.rparent()), result);
		}
		return result;
	}
};
//#endif


// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_TENSORBASISH_SEEN

//EOF.
