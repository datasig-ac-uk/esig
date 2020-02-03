/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai,
Greg Gyurkó and Arend Janssen.

Distributed under the terms of the GNU General Public License,
Version 3. (See accompanying file License.txt)

************************************************************* */

//  utils.h

// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_UTILSH_SEEN
#define DJC_COROPA_LIBALGEBRA_UTILSH_SEEN

/// Provides maps between lie<> and free_tensor<> instances.
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class maps
{
	/// The Free Associative Algebra Basis type
	typedef free_tensor_basis<SCA, RAT, n_letters, max_degree> TBASIS;
	/// The Free Lie Associative Algebra Basis type
	typedef lie_basis<SCA, RAT, n_letters, max_degree> LBASIS;
	/// The Free Lie Associative Algebra Basis KEY type
	typedef typename LBASIS::KEY LKEY;
	/// The Free Associative Algebra Basis KEY type
	typedef typename TBASIS::KEY TKEY;
	/// The Free Lie Associative Algebra element type
	typedef lie<SCA, RAT, n_letters, max_degree> LIE;
	/// The Free Associative Algebra element type
	typedef free_tensor<SCA, RAT, n_letters, max_degree> TENSOR;
public:
	/// Default constructor.
	maps(void) {}
public:

	/// computes the linear map
	class t2t
	{
		typedef alg::LET(*translator)(const LET);
		const translator h;
	public:
		t2t(translator arg) : h(arg) {}

		template <typename SCA1, typename RAT1, DEG n_letters1, DEG max_degree1>
		TENSOR	operator()(const alg::free_tensor<SCA1, RAT1, n_letters1, max_degree1>& in) const
		{
			typedef alg::free_tensor<SCA1, RAT1, n_letters1, max_degree1> TENSORIN;
			
			TENSOR out;
			for (typename TENSORIN::const_iterator it = in.begin(); it != in.end() ; ++it)
			{
				typename TENSOR::KEY y(it->first, h);
				if(SCA(0) == (out[y] += (it->second)))
					out.erase(y);
			}
			return out;
		}
	};

	/// Computes the free_tensor truncated exponential of a free lie element.
	inline TENSOR exp(LET l)
	{
		TKEY k; // empty word.
		TENSOR result(k);
		SCA coef(+1);
		DEG i;
		for (i = 1; i <= max_degree; ++i)
		{
			coef /= (RAT)i;
			k.push_back(l);
			result[k] = coef;
		}
		return result;
	}
	/// Returns the free_tensor corresponding to a free lie element.
	inline TENSOR l2t(const LIE& arg)
	{
		TENSOR result;
		typename LIE::const_iterator i;
		for (i = arg.begin(); i != arg.end(); ++i)
			result.add_scal_prod(expand(i->first), i->second);
		return result;
	}
	/// Returns the free lie element corresponding to a tensor_element.
	/**
	This is the Dynkin map obtained by right bracketing. Of course, the
	result makes sense only if the given free_tensor is the tensor expression
	of some free lie element.
	*/
	inline LIE t2l(const TENSOR& arg)
	{
		LIE result;
		typename TENSOR::const_iterator i;
		for (i = arg.begin(); i != arg.end(); ++i)
			result.add_scal_prod(rbraketing(i->first), i->second);
		typename LIE::iterator j;
		for (j = result.begin(); j != result.end(); ++j)
			(j->second) /= (RAT)(LIE::basis.degree(j->first));
		return result;
	}
	/// For a1,a2,...,an, return the expression [a1,[a2,[...,an]]].
	/**
	For performance reasons, the already computed expressions are stored in a
	static table to speed up further calculus. The function returns a
	constant reference to an element of this table.
	*/
	inline const LIE& rbraketing(const TKEY& k)
	{
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread
		boost::lock_guard<boost::recursive_mutex> lock(table_access);

		static std::map<TKEY, LIE> lies;
		typename std::map<TKEY, LIE>::iterator it;
		it = lies.find(k);
		if (it == lies.end())
			return lies[k] = _rbraketing(k);
		else
			return it->second;
	}
	/// Returns the free_tensor corresponding to the Lie key k.
	/**
	For performance reasons, the already computed expressions are stored in a
	static table to speed up further calculus. The function returns a
	constant reference to an element of this table.
	*/
	inline const TENSOR& expand(const LKEY& k)
	{
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread
		boost::lock_guard<boost::recursive_mutex> lock(table_access);

		static std::map<LKEY, TENSOR> table;
		typename std::map<LKEY, TENSOR>::iterator it;
		it = table.find(k);
		if (it == table.end())
			return table[k] = _expand(k);
		else
			return it->second;
	}
private:
	/// Computes recursively the free_tensor corresponding to the Lie key k.
	TENSOR _expand(const LKEY& k)
	{
		if (LIE::basis.letter(k))
			return (TENSOR)TENSOR::basis.keyofletter(LIE::basis.getletter(k));
		return commutator(expand(LIE::basis.lparent(k)),
			expand(LIE::basis.rparent(k)));
	}
	/// a1,a2,...,an is converted into [a1,[a2,[...,an]]] recursively.
	LIE _rbraketing(const TKEY& k)
	{
		if (TENSOR::basis.letter(k))
			return (LIE)LIE::basis.keyofletter(TENSOR::basis.getletter(k));
		return rbraketing(TENSOR::basis.lparent(k))
			* rbraketing(TENSOR::basis.rparent(k));
	}
};

/// Provides Campbell-Baker-Hausdorff formulas.
template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class cbh
{
	/// The Free Associative Algebra Basis type.
	typedef free_tensor_basis<SCA, RAT, n_letters, max_degree> TBASIS;
	/// The Free Lie Associative Algebra Basis type.
	typedef lie_basis<SCA, RAT, n_letters, max_degree> LBASIS;
	/// The Free Lie Associative Algebra Basis KEY type.
	typedef typename LBASIS::KEY LKEY;
	/// The Free Associative Algebra Basis KEY type.
	typedef typename TBASIS::KEY TKEY;
	/// The Free Lie Associative Algebra element type.
	typedef lie<SCA, RAT, n_letters, max_degree> LIE;
	/// The Free Associative Algebra element type.
	typedef free_tensor<SCA, RAT, n_letters, max_degree> TENSOR;
	/// The MAPS type.
	typedef maps<SCA, RAT, n_letters, max_degree> MAPS;
	/// Maps between lie and free_tensor instances.
	mutable MAPS m_maps;//TJL added mutable
public:
	/// The empty free_tensor.
	TENSOR empty_tensor;
	/// The empty free lie element.
	LIE empty_lie;
public:
	/// Default constructor.
	cbh(void) {}
public:
	/// Returns the CBH formula as a free lie element from a vector of letters.
	inline LIE basic(const std::vector<LET>& s) const
	{
		if (s.size() == 0) return empty_lie;
		TENSOR tmp(m_maps.exp(s[0]));
		typename std::string::size_type i;
		for (i = 1; i < s.size(); ++i)
			tmp *= m_maps.exp(s[i]);
		return m_maps.t2l(log(tmp));
	}
	/// Returns the CBH formula as a free lie element from a vector of lie.
	inline LIE full(const std::vector<LIE*>& lies) const
	{
		if (lies.size() == 0) return empty_lie;
		typename std::vector<LIE*>::size_type i;
		TENSOR tmp(exp(m_maps.l2t(*lies[0])));
		for (i = 1; i < lies.size(); ++i)
			tmp *= exp(m_maps.l2t(*lies[i]));
		return m_maps.t2l(log(tmp));
	}
};

// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_UTILSH_SEEN

//EOF.
