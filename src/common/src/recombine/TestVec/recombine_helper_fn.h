#ifndef recombine_helper_fn_h__
#define recombine_helper_fn_h__

#ifdef __cplusplus
extern "C"
{
#endif
	// recombine() 

	// operates on a buffer of "weights" which are positive doubles
	// and a buffer of an equal number of void pointers representing "points"
	
	// it needs a helper function that can take the buffer of pointers
	// and turn it into a buffer of doubles whose entries nd, nd+1, ..., n(d+1)-1 
	// is a d dimensional vector v[n] associated to pointer[n]

	// recombine, then identifies a subset of at most (d+1) pointers i_j and new weights W_j
	// so that \sum_j W_j v[i_j] = \sum_i w_i v[i]

	// In more detail, recombine is agnostic to point types, and resolves this through a 
	// Call Back Function that builds the vector co-ordinates of points
	// ARG1 points (in some way) to an indexed list of NoPointsToBeprocessed points to be reduced, and is const and typically NoPointsToBeprocessed pointers
	// ARG2 points to buffer of doubles length NoPointsToBeprocessed *(SmallestReducibleSetSize - 1)
	// ARG3 contains the size information and, if appropriate, preconditioning information that the expander function understands
	// the first two elements in the structure pointed to ARG3 and the implied dimensioned buffer in ARG2 have a fixed interpretation
	typedef void (*expander)(void*, double*, void*);

	// the structure pointed to by ARG3 must have initial segment
	struct CConditionedBufferHelper
	{
		size_t SmallestReducibleSetSize;
		size_t NoPointsToBeprocessed;
		void* pvCConditioning;
	};
	// and the buffers ARG1 and ARG2 must have at least the dimensions specified above.

	typedef CConditionedBufferHelper CBufferHelper;

	// an example of a conditioning that might be used in a given callback function
	//struct CConditioning
	//{
	//	double dMean;
	//	double dStdDev;
	//};

	//an example callback function
	// the void pointers point to "points" that are 1 dimensional real numbers represented as double's
	// the function aps each point x to the vector (1,x,x^2,....,x^(d-1))
	// the 1 is important as it ensures the mass is preserved
	// 
	//void ArrayOfpDoublesToVectorOfPowers1( void * pIn , double * pOut , void* vpCConditionedBufferHelper )
	//{
	//	CConditionedBufferHelper * pCConditionedBufferHelper = (CConditionedBufferHelper *) vpCConditionedBufferHelper;
	//	const size_t no_of_locations = pCConditionedBufferHelper -> NoPointsToBeprocessed;
	//	const size_t depth_of_vector = pCConditionedBufferHelper -> SmallestReducibleSetSize - 1;

	//	CConditioning * pConditioning = (CConditioning *) pCConditionedBufferHelper -> pvCConditioning ;
	//	const double dDisplacement = pConditioning -> dMean;
	//	const double dStdDev = pConditioning -> dStdDev;

	//	void ** pVoidIn = (void **) pIn; 
	//	for ( size_t i = 0 ; i < no_of_locations ; ++i , ++pVoidIn )
	//		for ( size_t j = 0 ; j < depth_of_vector ; ++j , ++pOut )
	//			if ( j < size_t(2) || (dDisplacement == double(0) && dStdDev == double(1)))
	//			{
	//				*pOut = pow( *((double *) (*pVoidIn)), (int)j );
	//			}
	//			else
	//			{
	//				*pOut = pow( (*((double *) (*pVoidIn)) - dDisplacement)/dStdDev , (int)j );
	//			}
	//}
#ifdef __cplusplus
}
#endif

#endif // recombine_helper_fn_h__

