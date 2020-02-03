/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai,
Greg Gyurkó and Arend Janssen.

Distributed under the terms of the GNU General Public License,
Version 3. (See accompanying file License.txt)

************************************************************* */

#pragma once
#include "implimentation_types.h"
#include "constlog2.h"

// VS2008 valid StaticAssert
template<bool> struct StaticAssert;
template<> struct StaticAssert<true> {};
#define STATIC_ASSERT(condition) do { StaticAssert<(condition)>(); } while(0)


template<size_t> struct intN;
template<>
struct intN<1>
{
	typedef int8_t myType;
};
template<>
struct intN<2>
{
	typedef int16_t myType;
};
template<>
struct intN<4>
{
	typedef int32_t myType;
};
template<>
struct intN<8>
{
	typedef int64_t myType;
};


template<size_t> struct uintN;
template<>
struct uintN<1>
{
	typedef uint8_t myType;
};
template<>
struct uintN<2>
{
	typedef uint16_t myType;
};
template<>
struct uintN<4>
{
	typedef uint32_t myType;
};
template<>
struct uintN<8>
{
	typedef uint64_t myType;
};

typedef double word_t;
//typedef float word_t;


/// helper structures to allow inline double manipulations
namespace {

	template<class real>
	struct fp_info
	{
		// integer types with same number of bits as real
		typedef  typename intN<sizeof(real)>::myType signed_int_type;
		typedef  typename uintN<sizeof(real)>::myType unsigned_int_type;

		//
		static const unsigned_int_type total_bits = sizeof(real) * std::numeric_limits<unsigned char>::digits;
		static const unsigned_int_type mantissa_bits_stored = std::numeric_limits<real>::digits - 1;
		static const unsigned_int_type exponent_bits = total_bits - (mantissa_bits_stored + 1);
		static const unsigned_int_type exponent_bias = (1 << (exponent_bits - 1)) - 1;
		static const unsigned_int_type exponent_mask = (((unsigned_int_type(1) << exponent_bits) - 1) << mantissa_bits_stored);
		static const unsigned_int_type mantissa_mask_zeroes = (((unsigned_int_type(1) << (exponent_bits + 1)) - 1) << mantissa_bits_stored);
		static const bool ieefp_to_int_as_expected;

		fp_info()
		{
			assert(ieefp_to_int_as_expected);
		}
	};
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::mantissa_bits_stored;
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::total_bits;
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::exponent_bits;
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::exponent_bias;
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::mantissa_mask_zeroes;
	template<class re> const typename fp_info<re>::unsigned_int_type fp_info<re>::exponent_mask;
	// put sanity tests below - this should work for all big endian fp with radix 2 and normalization with omitted 1
	template<class re> const bool fp_info<re>::ieefp_to_int_as_expected =
		(
			reinterpret_cast<const re&>(fp_info<re>::mantissa_mask_zeroes) == (std::numeric_limits<re>::infinity() * -1)
			&&
			reinterpret_cast<const re&>(fp_info<re>::exponent_mask) == (std::numeric_limits<re>::infinity())
			);

}
/// tjl 12/11/2017 A template to compute the number of words in the truncated basis
template <unsigned No_Letters, unsigned DEPTH>
struct NoWords
{
	enum
	{
		ans = No_Letters * NoWords < No_Letters, DEPTH - 1 >::ans + 1
	};
};

template <unsigned No_Letters>
struct NoWords<No_Letters, 0>
{
	enum
	{
		ans = 1
	};
};

/// Base class for tensor_basis
template <unsigned No_Letters, unsigned DEPTH>
class
	_tensor_basis
{
private:

	/// A private constructor from doubles
	_tensor_basis(const word_t base)
		: _word(base)
	{
	}

	/// A word_t that contains a word
	word_t _word;

	///The number of Bits in a letter
	static const unsigned uBitsInLetter = ConstLog2 < No_Letters -
		1 > ::ans + 1;
	static const long long uMaxSizeAlphabet = (1 << uBitsInLetter);
	static const unsigned uMaxWordLength = (unsigned)(fp_info<word_t>::mantissa_bits_stored / uBitsInLetter);
	//tjl 12/11/2017
	static const long long uMaxFeatureDimension = 1 << (uBitsInLetter * DEPTH);
	static const fp_info<word_t> sanity_check;

public:

	///Letter
	typedef alg::LET LET;

	///Constructor

	///Checks that the DEPTH does not exceed the Maximum word length.
	_tensor_basis(void)
		: _word((word_t)1.)
	{
		STATIC_ASSERT(DEPTH <= uMaxWordLength);
		//static_assert(DEPTH <= uMaxWordLength, "specified length of words in tensor basis exceeds size available ");
	}

    template <unsigned No_Letters2, unsigned DEPTH2, class translator>
	///Checks that the DEPTH does not exceed the Maximum word length.
	_tensor_basis(_tensor_basis<No_Letters2, DEPTH2> arg, translator h )
		: _word((word_t)1.)
	{
		STATIC_ASSERT(DEPTH <= uMaxWordLength);
		//static_assert(DEPTH <= uMaxWordLength, "specified length of words in tensor basis exceeds size available ");
		assert(arg.size() <= uMaxWordLength);
		for (; arg.size() > 0; arg = arg.rparent())
			*this = (*this * (_tensor_basis(h(arg.FirstLetter()))));
	}

	///Destructor
	~_tensor_basis(void)
	{
	}

	/// Concatenates two words
	inline _tensor_basis& push_back(const _tensor_basis& rhs)
	{
		STATIC_ASSERT(std::numeric_limits<word_t>::is_iec559 && std::numeric_limits<double>::has_denorm);

		word_t dPowerOfTwo = rhs._word;
		reinterpret_cast<fp_info<word_t>::unsigned_int_type&>(dPowerOfTwo) &= fp_info<word_t>::mantissa_mask_zeroes;
		_word = _word * dPowerOfTwo + rhs._word - dPowerOfTwo;
		return *this;
	}

	/// Concatenates two words
	inline _tensor_basis operator* (const _tensor_basis& rhs) const
	{
		STATIC_ASSERT(std::numeric_limits<word_t>::is_iec559 && std::numeric_limits<double>::has_denorm);

		word_t dPowerOfTwo = rhs._word;
		reinterpret_cast<fp_info<word_t>::unsigned_int_type&>(dPowerOfTwo) &= fp_info<word_t>::mantissa_mask_zeroes;
		return _word * dPowerOfTwo + rhs._word - dPowerOfTwo;
	}

	/// Compares two words
	inline bool operator < (const _tensor_basis & rhs) const
	{
		assert(size() <= DEPTH || size() == end().size());
		return _word < rhs._word;
	}
	inline bool operator > (const _tensor_basis & rhs) const
	{
		assert(size() <= DEPTH || size() == end().size());
		return rhs._word < _word ;
	}
	inline bool operator == (const _tensor_basis & rhs) const
	{
		assert(size() <= DEPTH || size() == end().size());
		return rhs._word == _word;
	}
	inline bool operator != (const _tensor_basis & rhs) const
	{
		assert(size() <= DEPTH || size() == end().size());
		return rhs._word != _word;
	}

    _tensor_basis(const LET uLetter)
		: _word(static_cast<word_t>(uMaxSizeAlphabet + (uLetter - 1) % uMaxSizeAlphabet))
	{
		assert(0 < uLetter && uLetter <= No_Letters && No_Letters <= uMaxSizeAlphabet);
	}

	/// gives the number of letters in _word
	inline unsigned size() const
	{	
		fp_info<word_t>::unsigned_int_type sz = (
			fp_info<word_t>::signed_int_type(
				(reinterpret_cast<const fp_info<word_t>::unsigned_int_type&>(_word) 
				& fp_info<word_t>::exponent_mask)
				>> fp_info<word_t>::mantissa_bits_stored) - fp_info<word_t>::exponent_bias) 
			/ uBitsInLetter;
#ifdef _DEBUG
		int iExponent;
		frexp(_word, &iExponent);
		assert((iExponent - 1) % uBitsInLetter == 0);
		assert(sz == ((iExponent - 1) / uBitsInLetter));
#endif
		return (unsigned int)sz;
	}

	/// Returns the first letter of a _tensor_basis as a letter.
	inline LET FirstLetter() const
	{
		const word_t dShiftPlus1(uMaxSizeAlphabet * 2);
		const word_t dShift(uMaxSizeAlphabet);
		//static const word_t dMinusShift = 1 / dShift;

		assert(size() > 0);
		int iExponent;
		word_t dMantissa = frexp(_word, &iExponent);
		word_t ans;
		modf(dMantissa * dShiftPlus1, &ans);
		return LET(ans - dShift) + 1;
	}

	/// Checks validity of a finite instance of _tensor_basis
	bool valid() const
	{
		if (DEPTH > uMaxWordLength) abort();
		if (this->_word == _tensor_basis()._word)
			return true;
		else
			return size() <= DEPTH && (FirstLetter() - 1 < No_Letters) &&
			rparent().valid();
	}

	//TJL 21/08/2012
	friend class _LET;
	struct _LET {
		_tensor_basis& m_parent;
		size_t m_index;
		_LET(const size_t index, _tensor_basis& parent)
			:m_parent(parent), m_index(index)
		{
		}

		operator LET()
		{
			word_t dBottom, dMiddle, dTop;
			int iExponent;
			word_t dTemp, dMantissa;
			dMantissa = frexp(m_parent._word, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent - (m_index + 1) *
				uBitsInLetter));
			dMiddle = modf(dTemp, &dTop);
			dTemp = dMiddle + (word_t)1.;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + uBitsInLetter));
			dBottom = modf(dTemp, &dMiddle);
			_tensor_basis middle(dMiddle);
			return  middle.FirstLetter(); //adds a one implicitly
		}

		bool operator <(DEG arg) const
		{
			return operator LET() < arg;
		}

		_LET& operator +=(const size_t i)
		{
			word_t dBottom, dMiddle, dTop;
			int iExponent;
			word_t dTemp, dMantissa;
			dMantissa = frexp(m_parent._word, &iExponent);
			dTemp = ldexp(dMantissa, int(
				iExponent - (m_index + 1) *
				uBitsInLetter)
			);
			dMiddle = modf(dTemp, &dTop);
			dTemp = (word_t)1. + dMiddle;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + 1 * uBitsInLetter));
			dBottom = modf(dTemp, &dMiddle);
			dTemp = (word_t)1. + dBottom;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + m_index * uBitsInLetter));
			modf(dTemp, &dBottom);
			_tensor_basis top(dTop), middle(dMiddle), bottom(dBottom),
				ans_tb;
			_tensor_basis newmiddle(LET(middle.FirstLetter() + i));
			ans_tb = (top * newmiddle * bottom);
			m_parent = ans_tb;
			return *this;
		}

		_LET& operator =(LET i)
		{
			word_t dBottom, dMiddle, dTop;

			int iExponent;
			word_t dTemp, dMantissa;
			dMantissa = frexp(m_parent._word, &iExponent);
			dTemp = ldexp(dMantissa, int(
				iExponent - (m_index + 1) *
				uBitsInLetter)
			);
			dMiddle = modf(dTemp, &dTop);
			dTemp = (word_t)1. + dMiddle;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + uBitsInLetter));
			dBottom = modf(dTemp, &dMiddle);
			dTemp = (word_t)1. + dBottom;
			dMantissa = frexp(dTemp, &iExponent);
			dBottom = ldexp(dMantissa, int(iExponent + m_index * uBitsInLetter));

			_tensor_basis top(dTop), middle(dMiddle), bottom(dBottom);
			_tensor_basis newmiddle(i);
			m_parent._word = (top*newmiddle*bottom)._word;
			return *this;
		}

		bool tt() const
		{
			word_t dBottom, dMiddle, dTop;
			int iExponent;
			word_t dTemp, dMantissa;
			dMantissa = frexp(m_parent._word, &iExponent);
			dTemp = ldexp(dMantissa, int(
				iExponent - (m_index + 1) *
				uBitsInLetter)
			);
			dMiddle = modf(dTemp, &dTop);
			dTemp = (word_t)1. + dMiddle;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + 1 * uBitsInLetter));
			dBottom = modf(dTemp, &dMiddle);
			dTemp = (word_t)1. + dBottom;
			dMantissa = frexp(dTemp, &iExponent);
			dTemp = ldexp(dMantissa, int(iExponent + m_index * uBitsInLetter));
			modf(dTemp, &dBottom);
			_tensor_basis top(dTop), middle(dMiddle), bottom(dBottom),
				ans_tb;
			ans_tb = (top*middle*bottom);
			return m_parent._word == ans_tb._word;

			/*
			//push_back
			frexp(rhs._word, &iExponent);
			word_t dPowerOfTwo = ldexp((word_t).5, iExponent);
			_word = (_word - 1) * dPowerOfTwo + rhs._word;
			*/
		}
	};

	//TJL 21/08/2012
	/// Treats the basis word as an array and returns a "letter" starting at the highest end of the used part of _word.
	inline _LET operator[](const size_t arg)
	{
		assert(arg < size());
		size_t rarg = size() - 1 - arg;
		return _LET(rarg, *this);
	}

	inline LET operator[](const size_t arg) const
	{
		assert(arg < size());
		size_t rarg = size() - 1 - arg;
		_tensor_basis temp(*this);
		return _LET(rarg, temp);
	}

	/// Returns the first letter of a _tensor_basis in a _tensor_basis.
	inline _tensor_basis lparent() const
	{
		const word_t dShiftPlus1(uMaxSizeAlphabet * 2);
		//static const word_t dShift(uMaxSizeAlphabet);
		//static const word_t dMinusShift = 1 / dShift;

		assert(size() > 0);
		int iExponent;
		word_t dMantissa = frexp(_word, &iExponent);
		word_t ans;
		modf(dMantissa * dShiftPlus1, &ans);
		return ans;
	}

	/// Returns the _tensor_basis which corresponds to the sub-word after the first letter.
	inline _tensor_basis rparent() const
	{
		static const word_t dShiftPlus1(uMaxSizeAlphabet * 2);
		//static const word_t dShift(uMaxSizeAlphabet);
		//static const word_t dMinusShift = 1 / dShift;

		assert(size() > 0);
		int iExponent;
		word_t dMantissa = frexp(_word, &iExponent);
		word_t ans;
		word_t dPowerOfTwo = ldexp((word_t).5, int(iExponent - uBitsInLetter));
		return (modf(dMantissa * dShiftPlus1, &ans) + (word_t)1.) * dPowerOfTwo;
	}

	static _tensor_basis end()
	{
		return std::numeric_limits<word_t>::infinity();
	}

	friend std::ostream & operator << (std::ostream & os, const _tensor_basis < No_Letters, DEPTH > & word)
	{
		int iExponent;
		unsigned count = word.size();
		word_t dNormalised = frexp(word._word, &iExponent) * 2. - (word_t)1.;
		os << "(";
		while (count > 0) {
			word_t letter;
			dNormalised = modf(dNormalised * uMaxSizeAlphabet, &letter);
			os << letter + (word_t)1.;
			--count;
			if (count != 0)
				os << ",";
		}
		return os << ")";
	}
	
	//tjl 12/11/2017
	/// a helper class for hashing the keys
	struct hash
	{
		enum
		{
			// NoKeys <= HashEnd is the full dense dimension of the tensor
			NoKeys = NoWords<No_Letters, DEPTH>::ans,
			// [HashBegin, HashEnd) is the integer range into which the tensor basis is hashed
			HashBegin = 0,
			HashEnd = ((uMaxFeatureDimension - 1) << 1) + 1
		};
		/// hashes a key injectively to an integer in the range [HashBegin, HashEnd)
		size_t operator ()(const _tensor_basis & key) const
		{
			assert((2.0 * double(std::numeric_limits<size_t>::max() / 2 + 1) > double(HashEnd)));
			return static_cast<size_t>(key._word);
		}
	};
};

