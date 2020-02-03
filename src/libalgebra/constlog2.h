/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




#pragma once
#ifndef ConstLog2_h__
#define ConstLog2_h__

/// A template for constructing integer constants

/// For \f$exp > 0\f$, produces the largest unsigned long long integer \f$ans\f$ such that
/// \f$2^(ans+1) > exp\f$ and \f$ans\f$ is smallest integer with this property.

template <unsigned long long exp>
struct ConstLog2
{
	enum
	{
		ans = ConstLog2 < exp /2 >::ans+1
	};
};

template <>
struct ConstLog2<1>
{
	enum
	{
		ans = 0
	};
};

template <>
struct ConstLog2<0>
{
	enum
	{
		ans = 0
	};
};
/// Test of ConstLog2 template
template <unsigned long long RANGE>
struct TestConstLog2
{
	enum
	{
		ans = ((RANGE >> ConstLog2<RANGE>::ans) == 1) && TestConstLog2< RANGE - 1 >::ans
	};
};

template <>
struct TestConstLog2<1>
{
	enum
	{
		ans = ((1 >> ConstLog2<1>::ans) == 1) && (ConstLog2<0>::ans == 0)
	};
};


#endif // ConstLog2_h__
