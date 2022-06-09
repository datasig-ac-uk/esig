#include "stdafx.h" //  pre-compiled header function
#include "TestVec/RdToPowers2.h"
#include "TestVec/recombine_helper_fn.h" // CBufferHelper pCBufferHelper
#include "TestVec/RdToPowers.h"          // CMultiDimensionalBufferHelper pConditioning
#include <assert.h>              // assert
#include "lib/SHOW.h"                // SHOW() macro
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>
#include <valarray>
//#include <omp.h>


typedef double SCA;
typedef SCA* PSCA;

#ifdef __cplusplus
extern "C"
{
#endif
//prodsswitch prodmethod = prodsswitch::Prods;
//prodsswitch prodmethod = prodsswitch::Prods_wei1;
prodsswitch prodmethod = prodsswitch::Prods_cheb;
}


namespace 
{
	// the number of commutative monomials of degree D in L letters //Product[(j + D)/j, {j, 1, L - 1}]
	size_t f(const size_t L, const size_t D)
	{
		size_t ans;
		if (L == 0 && D > 0)
			ans = 0;
		else {
			ans = 1;
			for (size_t j = 1; j < L; ++j) {
				ans *= (j + D);
				//_ASSERT(ans % j == 0);
				ans /= j;
			}
		}
		return ans;
	}

	template<size_t L, size_t D>
	struct MonomialsHelp{
		enum{
			f = (MonomialsHelp< L - 1, D >::f * ((L-1) + D))/(L-1)
		};
	};	

	template<size_t D>
	struct MonomialsHelp<1, D>{
		enum{
			f = 1
		};
	};

	template<size_t D>
	struct MonomialsHelp <0,D>{
		enum{
			f = 0
		};
	};

	template<>
	struct MonomialsHelp<0, 0>{
		enum{
			f = 1
		};
	};

	template <const size_t L, const size_t D> 
	size_t FF(void) {return MonomialsHelp< L + 1, D>::f;}

	// the number of commutative monomials of degree <= D in L letters //Product[(j + D)/j, {j, 1, L }]
	size_t F(const size_t L, const size_t D)
	{
		return f(L + 1, D);
	}

	template<size_t L> // the dimension of a point
	struct prods2
	{	
		inline void static prods(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv)
		{
			//std::cout << "\nEntering prods2 at level " << L << "\n";
			for ( ; k <= D ; ++k, val *= *ptv)
				prods2 <L - 1>::prods (now, val, k, D, ptv + 1);
		}
	};

	template<>
	struct prods2<1>
	{	
		inline void static prods(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv )
		{
			//std::cout << "\nEntering prods2 at level " << 1 << "\n";
			for ( ; k <= D ; ++k, val *= *ptv)
				*(now++) = val;
		}
	};

	void prods_test(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
	{
		if (ptv < end && k < D)
			for ( ; k <= D ; ((k++ < D) ? val *= *ptv : 0 ))
				prods_test(now, val, k, D, ptv + 1, end);
		else 
			*(now++) = val;
	}

	struct data{
		SCA v;
		size_t k;
		inline	friend bool operator < (const data& lhs, const data& rhs) {return rhs.k < lhs.k;}  //reversed
		inline	data&  operator += (const data& rhs) { v *= rhs.v ; k += rhs.k ; return *this ; }
		inline  SCA update(const SCA x_val) {  ++k ; v *= x_val ; return v ;}
		inline  SCA update2(const SCA x_val) { ++k ; return v * x_val ;}
	};

	// given i1<=...<=ij and the associated product vj if j<D choose ij+1 
	// given [i,L), D and val, cycle through j in [i,L) multiply val and decrement D call the function again
	// if gets to L or D = 0 output value
	// return after looping - job done
	//
	// 
	void prods_wei1(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
	{
		const size_t L = end - ptv;
		data default_data;
		default_data.v = SCA(1); 
		default_data.k = 0;
		data max_data;
		max_data.v = SCA(1); 
		max_data.k = D;

		*(now++) = SCA(1);
		std::vector<data> Z(L, default_data);

		std::vector<data>::iterator Zcurrent, Zb(std::begin(Z)), Ze(Zb + L);
		while ((Zcurrent = std::upper_bound(Zb, Ze, max_data)) != Ze)
		{
			*(now++) = Zcurrent->update(ptv[Zcurrent - Zb]);
			std::fill(Zb, Zcurrent, *Zcurrent);
		}
	}

	void prods_nonrecursive3(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
	{
		const size_t L = end - ptv;
		// [pvt,end) is a range of values we call the POINT
		// this code aims to produce the list [now,...) of all products of powers of these 
		// elements having total power at most D; we call this range the EVALUATION

		// as we enumerate these products we associate and maintain state variables
		// suppose [pvt,end) = x,y,z (or y_p)
		// the current product is x^l * y^n * z^m where l + n + m <= D and l, m, n are >= 0
		// we record 
		//            x^l * y^n * z^m, x^l * y^n, x^l (or Y_p) for p \in [0,L) 
		//        D,  l + n + m, l + n, l (or N_p) p \in [-1,L)
		// and using the graded reverse lex order we can iterate through all the values
		// using this state variable hence the calculations are in the most central part
		// of the cpu
		// the algorithm is to
		//
		// poutput, y_[0,L) and D are given
		// initialise Y_[0,L) and *(poutput++) to 1, N_[0,L) to 0 and N_(-1) to D 
		// while ((first p where the degree N_p is less than N_-1) != end)
		// multiply Y_p -> Y_p*y_p and write Y_p to *(poutput++)
		// fill N_[0,p) with N_p and then Y_[0,p) with Y_p 
		// endwhile
		//
		// (note there are order/pipelining implications here)

		//init
		data default_data;
		default_data.v = SCA(1); 
		default_data.k = 0;
		data max_data;
		max_data.v = SCA(1); 
		max_data.k = D;

		*(now++)=SCA(1);
		data Z[20];
		assert(L < 20);
		std::fill(Z, Z + L, default_data);
		data* Zcurrent;
		while ((Zcurrent = std::upper_bound(Z, Z + L, max_data)) != Z + L )
		{
			*(now++) = Zcurrent -> update (*(ptv + size_t(Zcurrent - Z)));
			std::fill(Z, Zcurrent, *Zcurrent);		
		}
	}

	void prods_nonrecursive2(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
	{
		const size_t L = end - ptv;
		// [pvt,end) is a range of values we call the POINT
		// this code aims to produce the list [now,...) of all products of powers of these 
		// elements having total power at most D; we call this range the EVALUATION

		// as we enumerate these products we associate and maintain state variables
		// suppose [pvt,end) = x,y,z (or y_p)
		// the current product is x^l * y^n * z^m where l + n + m <= D and l, m, n are >= 0
		// we record 
		//            x^l * y^n * z^m, x^l * y^n, x^l (or Y_p) for p \in [0,L) 
		//        D,  l + n + m, l + n, l (or N_p) p \in [-1,L)
		// and using the graded reverse lex order we can iterate through all the values
		// using this state variable hence the calculations are in the most central part
		// of the cpu
		// the algorithm is to
		//
		// poutput, y_[0,L) and D are given
		// initialise Y_[0,L) and *(poutput++) to 1, N_[0,L) to 0 and N_(-1) to D 
		// while ((first p where the degree N_p is less than N_-1) != end)
		// multiply Y_p -> Y_p*y_p and write Y_p to *(poutput++)
		// fill N_[0,p) with N_p and then Y_[0,p) with Y_p 
		// endwhile
		//
		// (note there are order/pipelining implications here)

		//init
		*(now++)=SCA(1);

		SCA Y[20];
		size_t N[20];
		assert(L < 20);
		std::fill(N, N + L, 0);
		std::fill(Y, Y + L, SCA(1));

		size_t* Ncurrent;
		while ((Ncurrent = std::upper_bound(N, N + L, D, std::greater<size_t>())) != N + L )
		{
			const size_t ofset = (Ncurrent - N);
			SCA* const Y2(Y + ofset);
			std::fill(N, Ncurrent, ++(*Ncurrent));
			*(now++) = (*(Y2) *= *(ptv + ofset));
			std::fill( Y, Y2, *(Y2) );		
		}
	}

	void prods_nonrecursive(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
	{
		// [pvt,end) is a range of values we call the POINT
		// this code aims to produce the list [now,...) of all products of powers of these 
		// elements having total power at most D; we call this range the EVALUATION

		// the empty monomial is always included
		*(now++)= SCA(1);

		// as we enumerate these products we associate and maintain state variables
		// suppose [pvt,end) = x,y,z (or y_p)
		// the current product is x^l * y^n * z^m where l + n + m <= D and l, m, n are >= 0
		// we record 
		//            x^l * y^n * z^m, x^l * y^n, x^l (or Y_p) for p \in [0,L) 
		//        D,  l + n + m, l + n, l (or N_p) p \in [-1,L)
		// and using the graded reverse lex order we can iterate through all the values
		// using this state variable hence the calculations are in the most central part
		// of the cpu
		// the algorithm is to
		//
		// poutput, y_[0,L) and D are given
		// initialise Y_[0,L) and *(poutput++) to 1, N_[0,L) to 0 and N_(-1) to D 
		// while ((first p where the degree N_p is less than N_-1) != end)
		// multiply Y_p -> Y_p*y_p and write Y_p to *(poutput++)
		// fill N_[0,p) with N_p and then Y_[0,p) with Y_p 
		// endwhile
		//
		// (note there are order/pipelining implications here)
		//

		size_t L = end - ptv;
		if (L!=0)
		{
			std::vector <size_t> ks( L, 0); // a monotone integer valued distribution function
			const std::vector < size_t >::iterator beginkit = ks.begin(), endkit = ks.end();

			std::vector <SCA> vs( L, SCA(1)); // the partial products of the monomial back to the root
			const std::vector < SCA >::iterator beginvit = vs.begin(), endvit = vs.end();

			for (size_t offset(0) ; offset != L ; 
				offset = (offset < L)?((*(beginkit + offset))++
				, std::fill_n(beginkit, offset, *(beginkit+offset))
				,*(now++) = ((*(beginvit + offset)) *= *(ptv + offset))
				, std::fill_n(beginvit, offset, *(beginvit+offset))
				, 0) : L
				)
			{
				for(; *beginkit < D;  ++(*beginkit), *(now++) = (*beginvit *= *ptv) );
				offset = std::upper_bound(beginkit, endkit, D, std::greater<size_t>()) - beginkit;
				//for( offset = L ; *(beginkit + offset - 1) != *beginkit ; --offset );
			}	
		}
		//// enumerate the monomials only
		//	for (size_t offset(0) ; offset != L ; 
		//     offset = (offset < L)?(*(rbeginit + offset))++, std::fill_n(rbeginit, offset, *(rbeginit+offset)), 0): L)
		//	{
		//		for(; *rbeginit < D;  ++(*rbeginit) );
		//		for( offset = L ;  *(rbeginit + offset - 1) != *rbeginit ; --offset ); // use an algo for this
		//	}

		// if degree less than D increase end value by 1
		// find the first smaller degree monomial
		// the degree is D so look for the first initial part of the monomial that has lower degree
		// increment the last part of that by one and extend value to the full length L
		// increment the last part of that initial monomial and extend value to the full length L

	}

}

void prods(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end)
{
	switch(prodmethod)
	{

	case prodsswitch::Prods2:
		prods2<4>::prods(now,val,k,D,ptv); //set the 4 appropriately 
		break;
	
	case prodsswitch::Prods_test:
		prods_test(now,val,k,D,ptv,end);
		break;
	
	case prodsswitch::Prods_nonrecursive3:
		prods_nonrecursive3(now,val,k,D,ptv,end);
		break;
	
	case prodsswitch::Prods_nonrecursive2:
		prods_nonrecursive2(now,val,k,D,ptv,end);
		break;

	case prodsswitch::Prods_nonrecursive:
		prods_nonrecursive(now,val,k,D,ptv,end);
		break;
	
	case prodsswitch::Prods_wei1:
		prods_wei1(now,val,k,D,ptv,end);
		break;

	case prodsswitch::Prods_cheb:
		// a recursive function
		// k the current degree of val
	{
		SCA  xval(*ptv), val_o(val), val_oo(xval * val);
		for (; k <= D; (((k++) < D) ? (val = 2. * xval * val_o - val_oo, val_oo = val_o, val_o = val) : (0)))

			// val is updated by *= the next value at the end of every loop
			// val is written only after considering all the letters
			// no action occurs if the degree has already been reached
			if (ptv + 1 < end)
			{
			prods(now, val, k, D, ptv + 1, end);
			}
			else
				(*(now++) = val);
	}
		break;

	case prodsswitch::Prods:
	default :
	{	// a recursive function
		// k the current degree of val
	 for ( ; k <= D ; (((k++ ) < D) ? (val *= (*ptv)) : (0) ))
		
		 // val is updated by *= the next value at the end of every loop
		 // val is written only after considering all the letters
		 // no action occurs if the degree has already been reached
			if (ptv + 1 < end)
			{
				prods(now, val, k, D, ptv + 1, end);
			}
			else 
				(*(now++) = val);
		}
	}
}

void RdToPowers(void* pIn, SCA* pOut, void* vpCBufferHelper)
{
	CBufferHelper* pCBufferHelper = (CBufferHelper*)vpCBufferHelper;
	const size_t no_of_locations = pCBufferHelper->NoPointsToBeprocessed;
	const size_t depth_of_vector = pCBufferHelper->SmallestReducibleSetSize - 1;

	CMultiDimensionalBufferHelper* pConditioning = (CMultiDimensionalBufferHelper*)pCBufferHelper->pvCConditioning;
	const size_t D = pConditioning->D;
	const size_t L = pConditioning->L;

	assert (depth_of_vector == F(L,D));

	//pIn is a null pointer to an array of null pointers, each of which points to sequences of L SCAs
	//void** pVoidIn = (void**)pIn; // a pointer to the first element of an array of null pointers

	std::vector<SCA> MAX(L, 0.), MIN(L, 0.), rescaled(L, 0.);
	std::valarray<SCA> buffer(0., L * no_of_locations);
	size_t j(0);

	for ( void** pVoidIn = (void**)pIn; pVoidIn < (void**)pIn + no_of_locations; ++pVoidIn)
	{	
		SCA* pInRecordBegin((SCA*)(*(pVoidIn)));
		for (size_t i = 0; i < L; ++i,++j)
		{
			MAX[i] = std::max(pInRecordBegin[i], MAX[i]);
			MIN[i] = std::min(pInRecordBegin[i], MIN[i]);
			buffer[j] = pInRecordBegin[i];
		}
	}	
	SCA* pOutBegin(pOut);		

#pragma omp parallel for
	for (ptrdiff_t j = 0; j < ptrdiff_t(no_of_locations); ++j)
	{
		SCA* now = pOutBegin + j * depth_of_vector;
		if(D==1)
		{
			now[0] = 1.;
			for (size_t i = 0; i < L; ++i)
				now[1 + i] = ((MAX[i] - MIN[i]) == 0.) ? 0. : (2 * buffer[i + L * j] - (MIN[i] + MAX[i])) / (MAX[i] - MIN[i]);
		}
		else
		{
			for (size_t i = 0; i < L; ++i)
				buffer[i + L * j] = ((MAX[i] - MIN[i]) == 0.) ? 0. : (2 * buffer[i + L * j] - (MIN[i] + MAX[i])) / (MAX[i] - MIN[i]);
			prods(now, SCA(1), 0, D, &buffer[L * j], &buffer[L * j] + L);
		}
	}
	pOutBegin += no_of_locations * depth_of_vector;
}
