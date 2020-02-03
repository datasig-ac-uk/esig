/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  algebra.h


// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_ALGEBRAH_SEEN
#define DJC_COROPA_LIBALGEBRA_ALGEBRAH_SEEN

/// A class to store and manipulate associative algebras elements.
/**
The template class BASIS must
(1) Satisfies the assumptions made by the sparse_vector template class.
(2) Provides two member functions
DEG BASIS::degree(const KEY&) const
BASIS::prod(const KEY&, const KEY&) with a return type suitable for
use as the first arg of sparse_vector::add_scal_prod(); it can be a key or a sparse vector for example
(3) The sparse_vector::MAP class must provide the swap() member function.
*/
template<class BASIS>
class algebra : public sparse_vector<BASIS>
{
public:
	/// The inherited sparse vector type.
	typedef sparse_vector<BASIS> VECT;
	/// Import of the iterator type from sparse_vector.
	typedef typename VECT::iterator iterator;
	/// Import of the constant iterator type from sparse_vector.
	typedef typename VECT::const_iterator const_iterator;
	/// Import of the KEY type from sparse_vector.
	typedef typename VECT::KEY KEY;
	/// Import of the SCALAR type from sparse_vector.
	typedef typename VECT::SCALAR SCALAR;	
	/// Import of the RATIONAL type from sparse_vector.
	typedef typename VECT::RATIONAL RATIONAL;	
	

	static const DEG MAX_DEGREE = BASIS::MAX_DEGREE;

	template <class Transform, size_t DEPTH1>
	inline void triangularbufferedmultiplyandcombine(const algebra& rhs, algebra& result, Transform fn) const
	{
		// create buffers to avoid unnecessary calls to MAP inside loop
		std::vector<std::pair<KEY, SCALAR> > buffer;
		std::vector<typename std::vector<std::pair<KEY, SCALAR> >::const_iterator>
			iterators;
		separate_by_degree(buffer, rhs, DEPTH1, iterators);

		typename std::vector<std::pair<KEY, SCALAR> >::const_iterator j, jEnd;
		const_iterator i(VECT::begin()), iEnd(VECT::end());
		for ( ; i != iEnd; ++i)
		{
			const KEY& k = i->first;
			size_t rhdegree = DEPTH1 - VECT::basis.degree(k);
			typename std::vector<std::pair<KEY, SCALAR> >:: const_iterator&
				jEnd = iterators[rhdegree];
			for (j = buffer.begin(); j != jEnd; ++j)
				result.add_scal_prod(VECT::basis.prod(i->first, j->first),
				fn(i->second * j->second));
		}

	}

	/// copy the (key, value) elements from rhs to a sorted vector buffer (using the key for sorting) 
	/// and construct an increasing vector iterators so that segment [iterators[i-1], iterators[i])
	/// contains keys of degree i; the first begins at [begin(), and the last ends at end), and it can be empty

	void separate_by_degree(std::vector<std::pair<KEY, SCALAR> > &buffer, const algebra &rhs, const size_t DEPTH1, std::vector<typename std::vector<std::pair<KEY, SCALAR> >::const_iterator> &iterators) const
	{
		buffer.assign(rhs.begin(), rhs.end());
#ifndef ORDEREDMAP
		std::sort(buffer.begin(), buffer.end(),
			[](const std::pair<KEY, SCALAR>&lhs, const std::pair<KEY, SCALAR>&rhs)->bool
		{return lhs.first < rhs.first; }
		);
#endif // ORDEREDMAP
 
		iterators.assign(DEPTH1 + 1, buffer.end());
		unsigned deg = 0;
		for (typename std::vector<std::pair<KEY, SCALAR> >::const_iterator j0 = buffer.begin();
			j0 != buffer.end();
			j0++)
		{
			DEG d = VECT::basis.degree(j0->first);
			assert(d >= deg && d <= DEPTH1); // order assumed to respect degree
			while (deg < d)
				iterators[deg++] = j0;
			// deg == d
		}
	}

	template <class Transform>
	inline void squarebufferedmultiplyandcombine(const algebra& rhs, algebra& result, Transform fn) const
	{	
		// create buffer to avoid unnecessary calls to MAP inside loop
		std::vector<std::pair<KEY, SCALAR> > buffer(rhs.begin(), rhs.end());
		const_iterator i;

		// DEPTH1 == 0
		typename std::vector<std::pair<KEY, SCALAR> >:: const_iterator j;
		for (i = VECT::begin(); i != VECT::end(); ++i)
		{
			for (j = buffer.begin(); j != buffer.end(); ++j)
				result.add_scal_prod(VECT::basis.prod(i->first, j->first),
				fn(i->second * j->second));
		}

	}

	// Transformations

	/// function objects for doing nothing to a scalar
	struct scalar_passthrough
	{
		SCALAR operator()(const SCALAR& arg)
		{
			return arg;
		}
	};
	/// function object for changing the sign of a scalar
	struct scalar_minus
	{
		SCALAR operator()(const SCALAR& arg)
		{
			return -arg;
		}
	};
	/// function object for post-multiplying a scalar by a scalar
	struct scalar_post_mult
	{
	private:
		SCALAR mFactor;
	public:
		scalar_post_mult(const SCALAR Factor = VECT::one)
			: mFactor(Factor) {}
		SCALAR operator()(const SCALAR arg)
		{
			return arg * mFactor;
		}
	};
	/// function object for post-multiplying a scalar by stored version of the scalar 1 / rational
	struct rational_post_div
	{
	private:
		SCALAR mFactor;
	public:
		rational_post_div(const RATIONAL Factor = VECT::one)
			: mFactor(VECT::one / Factor) {}
		SCALAR operator()(const SCALAR arg)
		{
			return arg * mFactor;
		}
	};

	// used to index types
	template<unsigned int T> 
	struct identity {}; 


	/// multiplies *this and rhs adding it to result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandadd(const algebra& rhs, algebra& result) const
	{
		bufferedmultiplyandadd( rhs, result, identity<DEPTH1>());
	}
private:
	/// multiplies *this and rhs adding it to result with optimizations coming from degree
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandadd(const algebra& rhs, algebra& result, identity<DEPTH1>) const
	{
		scalar_passthrough fn;
		triangularbufferedmultiplyandcombine<scalar_passthrough, DEPTH1>(rhs, result, fn);
	}
	/// multiplies *this and rhs adding it to result without optimizations coming from degree
	inline void bufferedmultiplyandadd(const algebra& rhs, algebra& result, identity<0>) const
	{
		scalar_passthrough fn;
		squarebufferedmultiplyandcombine(rhs, result, fn);
	}
public:
	/// multiplies *this and rhs subtracting it from result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsub(const algebra& rhs, algebra& result) const
	{
		bufferedmultiplyandsub(rhs, result, identity<DEPTH1>());
	}
private:
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsub(const algebra& rhs, algebra& result, identity<DEPTH1>) const
	{
		scalar_minus fn;
		triangularbufferedmultiplyandcombine<scalar_minus, DEPTH1>(rhs, result, fn);
	}

	/// multiplies *this and rhs subtracting it to result without optimizations coming from degree
	inline void bufferedmultiplyandsub(const algebra& rhs, algebra& result, identity<0>) const
	{
		scalar_minus fn;
		squarebufferedmultiplyandcombine(rhs, result, fn);
	}
public:
	struct wrapscalar
	{
		const SCALAR& hidden;
		wrapscalar(const SCALAR& s)
			: hidden(s) {}
	};
	struct wraprational
	{
		const RATIONAL& hidden;
		wraprational(const RATIONAL& s)
			: hidden(s) {}
	};
	/// multiplies  *this and rhs adds it * s to result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsmult(const wrapscalar& ss, const algebra& rhs, algebra& result) const
	{
		bufferedmultiplyandsmult(ss, rhs, result, identity<DEPTH1>());
	}
private:
	/// multiplies  *this and rhs adds it * s to result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsmult(const wrapscalar& ss, const algebra& rhs, algebra& result, identity<DEPTH1>) const
	{
		scalar_post_mult fn(ss.hidden);
		triangularbufferedmultiplyandcombine<scalar_post_mult, DEPTH1>(rhs, result, fn);
	}

	/// multiplies *this and rhs adds it * s to result without optimizations coming from degree
	inline void bufferedmultiplyandsmult(const wrapscalar& ss, const algebra& rhs, algebra& result, identity<0>) const
	{
		scalar_post_mult fn(ss.hidden);
		squarebufferedmultiplyandcombine(rhs, result, fn);
	}
public:
	/// multiplies  *this and rhs adds it * s to result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsdiv(const algebra& rhs, const wraprational& ss, algebra& result) const
	{
		bufferedmultiplyandsdiv(rhs, ss, result, identity<DEPTH1>());
	}
private:
	/// multiplies  *this and rhs adds it * s to result
	template <unsigned DEPTH1>
	inline void bufferedmultiplyandsdiv(const algebra& rhs, const wraprational& ss, algebra& result, identity<DEPTH1>) const
	{
		rational_post_div fn(ss.hidden);
		triangularbufferedmultiplyandcombine<rational_post_div, DEPTH1>(rhs, result, fn);
	}

	/// multiplies *this and rhs adds it * s to result without optimizations coming from degree
	inline void bufferedmultiplyandsdiv(const algebra& rhs, const wraprational& ss, algebra& result,  identity<0>) const
	{
		rational_post_div fn(ss.hidden);
		squarebufferedmultiplyandcombine(rhs, result, fn);
	}
public:
public:
	/// Default constructor. 
	/**
	Constructs an empty algebra element.
	*/
	algebra(void) {}
	/// Copy constructor.
	algebra(const algebra& a)
		: VECT(a) {}
	/// Constructs an algebra instance from a sparse_vector.
	algebra(const VECT& v)
		: VECT(v) {}
	/// Unidimensional constructor.
	explicit algebra(const KEY& k, const SCALAR& s = VECT::one)
		: VECT(k, s) {}
public:	
	/// Multiplies the instance with scalar s.
	inline algebra& operator*=(const SCALAR& s)
	{
		VECT::operator *= (s);
		return *this;
	}
	/// Divides the instance by scalar s.
	inline algebra& operator/=(const RATIONAL& s)
	{
		VECT::operator /= (s);
		return *this;
	}
	/// Ensures that the return type is an instance of algebra.
	inline __DECLARE_BINARY_OPERATOR(algebra,*,*=,SCALAR)
		/// Ensures that the return type is an instance of algebra.
		inline __DECLARE_BINARY_OPERATOR(algebra,/,/=,SCALAR)
		/// Ensures that the return type is an instance of algebra.
		inline __DECLARE_BINARY_OPERATOR(algebra,+,+=,algebra)
		/// Ensures that the return type is an instance of algebra.
		inline __DECLARE_BINARY_OPERATOR(algebra,-,-=,algebra)
		/// Ensures that the return type is an instance of algebra.
		inline __DECLARE_UNARY_OPERATOR(algebra,-,-,VECT);
	/// Multiplies the instance by an instance of algebra.
	inline algebra& operator*=(const algebra& rhs)
	{
		algebra result;
		bufferedmultiplyandadd<MAX_DEGREE>(rhs, result);
		this->swap(result);
		return *this;
	}
	/// Binary version of the product of algebra instances.
	inline __DECLARE_BINARY_OPERATOR(algebra,*,*=,algebra);
	/// Adds to the instance a product of algebra instances.
	inline algebra& add_mul(const algebra& a, const algebra& b)
	{
		a.bufferedmultiplyandadd<MAX_DEGREE>(b, *this);
		return *this;
	}
	/// Subtracts to the instance a product of algebra instances.
	inline algebra& sub_mul(const algebra& a, const algebra& b)
	{
		a.bufferedmultiplyandsub<MAX_DEGREE>(b, *this);
		return *this;
	}
	/// Multiplies the instance by (algebra instance)*s.
	inline algebra& mul_scal_prod(const algebra& rhs, const SCALAR& s)
	{
		algebra result;
		bufferedmultiplyandsmult(rhs, wrapscalar(s), result);
		this->swap(result);
		return *this;
	}
	/// Multiplies the instance by (algebra instance)/s.
	inline algebra& mul_scal_div(const algebra& rhs, const RATIONAL& s)
	{
		algebra result;
		bufferedmultiplyandsdiv<MAX_DEGREE>(rhs, wraprational(s), result);
		this->swap(result);
		return *this;
	}
	/// Returns an instance of the commutator of two algebra instances.
	inline friend algebra commutator(const algebra& a, const algebra& b)
	{ // Returns a * b - b * a
		algebra result;
		a.bufferedmultiplyandadd<MAX_DEGREE>(b, result);
		b.bufferedmultiplyandsub<MAX_DEGREE>(a, result);
		return result;
	}
	/// Returns a truncated version of the instance, by using basis::degree().
	inline algebra truncate(const DEG min, const DEG max) const
	{
		algebra result;
		const_iterator i;
		for (i = VECT::begin(); i != VECT::end(); ++i)
			if ((VECT::basis.degree(i->first) >= min) && (VECT::basis.degree(i->first) <= max))
				result[i->first] = i->second;
		return result;
	}
	/// Returns the degree of the instance by using basis:degree()
	inline DEG degree(void) const
	{
		DEG result(0);
		const_iterator i;
		for (i = VECT::begin(); i != VECT::end(); ++i)
			result = std::max(result, VECT::basis.degree(i->first));
		return result;
	}
};

// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_ALGEBRAH_SEEN

//EOF.
