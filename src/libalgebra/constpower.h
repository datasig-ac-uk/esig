/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai,
Greg Gyurkó and Arend Janssen.

Distributed under the terms of the GNU General Public License,
Version 3. (See accompanying file License.txt)

************************************************************* */

#pragma once
#ifndef ConstPower_h__
#define ConstPower_h__

/// A template for constructing integer constants

/// ConstPower< arg, exp>::ans is the constant integer value of arg^exp
template <unsigned long long arg, alg::DEG exp>
struct ConstPower
{
	enum //: unsigned long long
	{
		ans = ConstPower < arg, exp / 2 >::ans * ConstPower < arg, exp / 2 >::ans * ((exp % 2 == 0) ? 1 : arg),
		enum1_force_long_long = 18446744073709551615ULL
	};
};

template < unsigned long long arg>
struct ConstPower < arg, 0>
{
	enum //: unsigned long long
	{
		ans = 1,
		enum1_force_long_long = 18446744073709551615ULL
	};
};

/// Test of ConstPower Template

/// TestConstPower<3,5> tests the ConstPower Template for some the powers x^y with x < 4 and y < 16
template < unsigned long long arg, alg::DEG exp >
struct TestConstPower
{
	enum //: unsigned long long
	{
		intermediate = ConstPower < arg, exp>::ans * TestConstPower<arg, exp - 1 >::intermediate,
		ans = (intermediate == ConstPower < arg, (ConstPower < exp, 2 >::ans + exp) / 2>::ans) && TestConstPower<arg - 1,
		exp >::ans,
		enum1_force_long_long = 18446744073709551615ULL
	};
};

template < unsigned long long arg >
struct TestConstPower < arg, 0 >
{
	enum //: unsigned long long
	{
		intermediate = ConstPower < arg, 0 >::ans,
		ans = (intermediate == 1),
	    enum1_force_long_long = 18446744073709551615ULL
	};
};

template < alg::DEG exp >
struct TestConstPower < 1, exp >
{
	enum
	{
		intermediate = ConstPower < 1, exp >::ans,
		ans = (intermediate == 1)
	};
};

template <>
struct TestConstPower < 1, 0 >
{
	enum
	{
		intermediate = ConstPower < 1, 0 >::ans,
		ans = (intermediate == 1)
	};
};

#endif // ConstPower_h__
