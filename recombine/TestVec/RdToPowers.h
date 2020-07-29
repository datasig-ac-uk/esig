#ifndef RdToPowers_h__
#define RdToPowers_h__

#ifdef __cplusplus
extern "C"
{
#endif

	void RdToPowers(void* pIn, double* pOut, void* vpCBufferHelper);
	// the structure pointed to be arg3 must have intial segment
	//struct CConditionedBufferHelper
	//{
	//	size_t SmallestReducibleSetSize;
	//	size_t NoPointsToBeprocessed;
	//	void* pvCConditioning;
	//};
	// and the buffers ARG1 and ARG2 must have at least the dimensions specified above.

	// an example of a conditioning that might be used in a given callback function
	struct CMultiDimensionalBufferHelper
	{
		// all commutative monomials of degree <= D in L letters
		size_t L;
		size_t D;
	};

#ifdef __cplusplus
}
#endif

#endif // RdToPowers_h__
