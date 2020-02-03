/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */



//  gmpwrapper.h
//  C++ class wrapper for GMP types.


// Include once wrapper
#ifndef DJC_COROPA_LIBUTILS_GMPWRAPPERH_SEEN
#define DJC_COROPA_LIBUTILS_GMPWRAPPERH_SEEN

#ifdef __GNU__
#pragma warning(push)
#pragma warning(disable : 4800)
#include <gmpxx.h>
#pragma warning(pop)
#else
#include <stdio.h>
#include <iostream>
#include <string>
#include <gmp.h>

#ifdef _WIN64 
typedef unsigned long long int  mp_bitcnt_t; 
#else 
typedef unsigned long int mp_bitcnt_t; 
#endif 


class mpq_class
{
private:
	mpq_t mp;
public:
	mp_bitcnt_t get_prec() const
	{
		return mpf_get_default_prec();
	}
	mpq_class()
	{
		mpq_init(mp);
	}
	mpq_class(const mpq_class& z)
	{
		mpq_init(mp);
		mpq_set(mp, z.mp);
	}
	mpq_class(signed char c)
	{
		mpq_init(mp);
		mpq_set_si(mp, c, 1);
	}
	mpq_class(unsigned char c)
	{
		mpq_init(mp);
		mpq_set_ui(mp, c, 1);
	}
	mpq_class(signed int i)
	{
		mpq_init(mp);
		mpq_set_si(mp, i, 1);
	}
	mpq_class(unsigned int i)
	{
		mpq_init(mp);
		mpq_set_ui(mp, i, 1);
	}
	mpq_class(signed short int s)
	{
		mpq_init(mp);
		mpq_set_si(mp, s, 1);
	}
	mpq_class(unsigned short int s)
	{
		mpq_init(mp);
		mpq_set_ui(mp, s, 1);
	}
	mpq_class(signed long int l)
	{
		mpq_init(mp);
		mpq_set_si(mp, l, 1);
	}
	mpq_class(unsigned long int l)
	{
		mpq_init(mp);
		mpq_set_ui(mp, l, 1);
	}
	mpq_class(float f)
	{
		mpq_init(mp);
		mpq_set_d(mp, f);
	}
	mpq_class(double d)
	{
		mpq_init(mp);
		mpq_set_d(mp, d);
	}
	explicit mpq_class(const char* s)
	{
		mpq_set_str(mp, s, 0);
	}
	mpq_class(const char* s, int base)
	{
		mpq_set_str(mp, s, base);
	}
	explicit mpq_class(const std::string& s)
	{
		mpq_set_str(mp, s.c_str(), 0);
	}
	mpq_class(const std::string& s, int base)
	{
		mpq_set_str(mp, s.c_str(), base);
	}
	explicit mpq_class(mpq_srcptr z)
	{
		mpq_init(mp);
		mpq_set(mp, z);
	}
	~mpq_class()
	{
		mpq_clear(mp);
	}
	inline mpq_class& operator=(const mpq_class& z)
	{
		mpq_set(mp, z.mp);
		return *this;
	}
	inline mpq_class& operator=(signed char c)
	{
		mpq_set_si(mp, c, 1);
		return *this;
	}
	inline mpq_class& operator=(unsigned char c)
	{
		mpq_set_ui(mp, c, 1);
		return *this;
	}
	inline mpq_class& operator=(signed int i)
	{
		mpq_set_si(mp, i, 1);
		return *this;
	}
	inline mpq_class& operator=(unsigned int i)
	{
		mpq_set_ui(mp, i, 1);
		return *this;
	}
	inline mpq_class& operator=(signed short int s)
	{
		mpq_set_si(mp, s, 1);
		return *this;
	}
	inline mpq_class& operator=(unsigned short int s)
	{
		mpq_set_ui(mp, s, 1);
		return *this;
	}
	inline mpq_class& operator=(signed long int l)
	{
		mpq_set_si(mp, l, 1);
		return *this;
	}
	inline mpq_class& operator=(unsigned long int l)
	{
		mpq_set_ui(mp, l, 1);
		return *this;
	}
	inline mpq_class& operator=(float f)
	{
		mpq_set_d(mp, f);
		return *this;
	}
	inline mpq_class& operator=(double d)
	{
		mpq_set_d(mp, d);
		return *this;
	}
	inline mpq_class& operator=(const char* s)
	{
		mpq_set_str(mp, s, 0);
		return *this;
	}
	inline mpq_class& operator=(const std::string& s)
	{
		mpq_set_str(mp, s.c_str(), 0);
		return *this;
	}
	inline std::string get_str(int base = 10) const
	{
		std::string temp(mpq_get_str(0, base, mp));
		return temp;
	}
	inline mpq_class operator-(void) const
	{
		mpq_class result;
		mpq_neg(result.mp, mp);
		return result;
	}
	inline mpq_class operator*(const mpq_class& rhs) const
	{
		mpq_class result;
		mpq_mul(result.mp, mp, rhs.mp);
		return result;
	}
	inline mpq_class operator/(const mpq_class& rhs) const
	{
		mpq_class result;
		mpq_div(result.mp, mp, rhs.mp);
		return result;
	}
	inline mpq_class operator-(const mpq_class& rhs) const
	{
		mpq_class result;
		mpq_sub(result.mp, mp, rhs.mp);
		return result;
	}
	inline mpq_class operator+(const mpq_class& rhs) const
	{
		mpq_class result;
		mpq_add(result.mp, mp, rhs.mp);
		return result;
	}
	inline mpq_class& operator+=(const mpq_class& rhs)
	{
		mpq_add(mp, mp, rhs.mp);
		return *this;
	}
	inline mpq_class& operator-=(const mpq_class& rhs)
	{
		mpq_sub(mp, mp, rhs.mp);
		return *this;
	}
	inline mpq_class& operator*=(const mpq_class& rhs)
	{
		mpq_mul(mp, mp, rhs.mp);
		return *this;
	}
	inline mpq_class& operator/=(const mpq_class& rhs)
	{
		mpq_div(mp, mp, rhs.mp);
		return *this;
	}
	inline bool operator==(const mpq_class& rhs) const
	{
		return (mpq_equal(mp, rhs.mp) != 0);
	}
	inline bool operator!=(const mpq_class& rhs) const
	{
		return !operator == (rhs);
	}
	inline bool operator>(const mpq_class& rhs) const
	{
		return mpq_cmp(mp, rhs.mp) > 0;
	}
	inline bool operator<(const mpq_class& rhs) const
	{
		return mpq_cmp(rhs.mp, mp) > 0;
	}
	inline friend mpq_class abs(const mpq_class& arg)
	{
		mpq_class result;
		mpq_abs(result.mp, arg.mp);
		return result;
	}
	inline friend std::ostream& operator<<(std::ostream& o, const mpq_class& z)
	{
		return o << z.get_str();
	}
};

#endif // __GNU__ check.

#endif //DJC_COROPA_LIBUTILS_GMPWRAPPERH_SEEN

