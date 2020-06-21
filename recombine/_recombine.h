#pragma once
// interface headers:
#include <stddef.h>             // size_t ptrdiff_t
#ifdef __cplusplus
extern "C"
{
#endif
	void _recombineC(
		size_t stCubatureDegree
		, ptrdiff_t dimension
		, ptrdiff_t no_locations
		, ptrdiff_t* pno_kept_locations
		, const void** ppLocationBuffer
		, double* pdWeightBuffer
		, size_t* KeptLocations
		, double* NewWeights
	);

#ifdef __cplusplus
} // extern "C"
#endif
