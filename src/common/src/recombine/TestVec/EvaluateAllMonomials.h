#pragma once

// evaluates all monomials (in a fixed enumeration) on a given point of arbitrary no of slots
// assuming L slots, and monomials of degree D.

// a 6 dimensional cubature of 6th order needs 924 degrees of freedom and typically 924 ^ 3 doubles takes about 6GB
// a 6 dimensional cubature of 4th order needs 210 degrees of freedom and typically 210 ^ 3 doubles takes about 70MB
// probably about 100 times faster
//
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#else
#define _ASSERT(expr) ((void)0)

#define _ASSERTE(expr) ((void)0)
#endif
#include <algorithm>
#include <cmath>
#include "OstreamContainerOverloads.h"



class EvaluateAllMonomials
{
	size_t mNoSlots;
	size_t mNotAMonomialOfThisDegree;
	size_t mNoProducts;
	std::vector<std::vector<double> > samples; // for each degree, and ordered set of samples

	std::vector<double> join(const std::vector<double>& lhs, const std::vector<double>& rhs) const
	{
		std::vector<double> ans;
		size_t lh(lhs.size()), rh(rhs.size());
		ans.resize(lh * rh);
		for (size_t i = 0; i < lh; ++i)
			for (size_t j = 0; j < rh; ++j)
				ans[i * rh + j] = lhs[i] * rhs[j];
		return ans;
	}

	// How many Products are there with given degree
	// f( L , D ) = \sum_{j=0...D} f( L-k , j ) *  f( k , D-j )
	// f( L , D ) = \sum_{j=0...D} f( L-1 , j ) *  f( 1 , D-j )
	// f( L , D ) = \sum_{j=0...D} f( L-1 , j ) *  1
	// f( 0 , D ) = 1 if D = 0, f( 0 , D ) = 0 otherwise
	// f( 1 , D ) = 1
	// f( L , D ) = (1 + D) (2 + D) (3 + D) (4 + D) ... ((L-1) + D)/(L-1)!
	//            = Product[(j + D)/j, {j, 1, L - 1}]

	// the sum of all these dimensions has a virtually identical form:
	// F(L,D) = \sum_{0 <= k <= D} f(L,k) 
	//        = Product[(j + D)/j, {j, 1, L}]
	//        = f(L + 1, D)
	//
	// the empty evaluation 1 is always provided
	//

	// the number of commutative monomials of degree D in L letters //Product[(j + D)/j, {j, 1, L - 1}]
	static size_t f(size_t L, size_t D)
	{
		size_t ans;
		if (L == 0 && D > 0)
			ans = 0;
		else
		{
			ans = 1;
			for (size_t j = 1; j < L; ++j)
			{
				ans *= (j + D);
				_ASSERT(ans % j == 0);
				ans /= j;
			}
		}
		return ans;
	}



public:

	// the number of commutative monomials of degree <= D in L letters //Product[(j + D)/j, {j, 1, L }]
	static size_t F(size_t L, size_t D)
	{
		return f(L + 1, D);
	}

	size_t NoProducts() const
	{
		return mNoProducts;
	}
//#pragma warning(disable:4996)
//	void vdMonomialBuffer(double* pOut, double* pOutEnd)
//	{
//		for (size_t i = 0; i < mNotAMonomialOfThisDegree; ++i)
//			pOut = std::copy(samples[i].begin(), samples[i].end(), pOut);
//	}
#pragma warning(default:4996)
	EvaluateAllMonomials(const std::vector<double>& point, const size_t Degree = 0)
		: mNoSlots(point.size()),
		  mNotAMonomialOfThisDegree(Degree + 1),
		  mNoProducts(0)
	{
		size_t point_dim(point.size());

		if (0 == point_dim)
		{
			samples.resize(1);
			samples[0].resize(1);
			samples[0][0] = double(1);
			mNoProducts = 1;
		}
		else
		{
			if (1 == point_dim)
			{
				samples.resize(mNotAMonomialOfThisDegree);
				for (size_t i = 0; i < mNotAMonomialOfThisDegree; ++i)
				{
					samples[i].resize(1);
					_ASSERT(i == size_t((int)i));
					samples[i][0] = pow(point[0], (int)i);
				}
				mNoProducts = mNotAMonomialOfThisDegree;
			}
			else
			{

				size_t point_dim_lhs(point_dim / 2);


				std::vector<double> LHpoint(&point[0],&point[0] + point_dim_lhs);
				std::vector<double> RHpoint(&point[0] + point_dim_lhs, &point[0] + point_dim);
				EvaluateAllMonomials LHS(LHpoint, Degree);
				EvaluateAllMonomials RHS(RHpoint, Degree);

				samples.resize(mNotAMonomialOfThisDegree);
				//SHOW(mNoSlots);
				//SHOW(Degree);
				for (size_t i = 0; i < mNotAMonomialOfThisDegree; ++i)
				{
					for (size_t j = 0; j <= i; ++j)
					{
						std::vector<double> temp = join(LHS.samples[j], RHS.samples[i - j]);
						samples[i].insert(samples[i].end(), temp.begin(), temp.end());
					}
					mNoProducts += samples[i].size();
					//SHOW(i);
					//SHOW(mNoSlots);
					//SHOW(f(mNoSlots, i));
					//SHOW(samples[i]);
				}
			}
		}
	}

	~EvaluateAllMonomials(void) {}
};
