// private headers
#include "recombine/recombine.h"
#include "TestVec/RdToPowers.h" // CMultiDimensionalBufferHelper
#include "TestVec/EvaluateAllMonomials.h" //EvaluateAllMonomials::F
#include "_recombine.h"
#include <vector>

void _recombineC(size_t stCubatureDegree, ptrdiff_t dimension, ptrdiff_t no_locations, ptrdiff_t* pno_kept_locations, const void** ppLocationBuffer, double* pdWeightBuffer, size_t* KeptLocations, double* NewWeights)
{
	ptrdiff_t& no_kept_locations = *pno_kept_locations;
	// the required max size of the out buffer needs to be known in advance by the calling routine
	size_t iNoDimensionsToCubature = EvaluateAllMonomials::F(dimension, stCubatureDegree);
	if (0 == no_locations)
	{
		no_kept_locations = iNoDimensionsToCubature;
		return;
	}

	// set up the input structure for conditioning the helper function
	CMultiDimensionalBufferHelper sConditioning;
	sConditioning.D = stCubatureDegree;
	sConditioning.L = dimension;

	// set up the input structure for data reduction "in"
	sCloud in;

	// chain optional extension information used to condition the data
	in.end = &sConditioning;

	// place the sizes of the buffers and their locations into the structure "in"
	in.NoActiveWeightsLocations = no_locations;
	in.LocationBuf = ppLocationBuffer;
	in.WeightBuf = pdWeightBuffer;

	// set up the output structure for data reduction "out"
	sRCloudInfo out;
	out.end = 0;

	// set the locations of these buffers into the structure "out"
	out.KeptLocations = KeptLocations;
	out.NewWeightBuf = NewWeights;

	// check the sizes of the out buffers
	if (*pno_kept_locations < iNoDimensionsToCubature)
	{
		*pno_kept_locations = 0;
		return;
	}
	// buffers reported big enough
	out.No_KeptLocations = iNoDimensionsToCubature;

	// setup the Recombine Interface data which will join the input and output
	sRecombineInterface data;
	data.end = 0;

	// bind in and out together in data
	data.pInCloud = &in;
	data.pOutCloudInfo = &out;

	// add the degree of the vectors used and the callback function that expands
	// the array of pointers to points into a long buffer of vectors
	data.degree = iNoDimensionsToCubature;

	data.fn = &RdToPowers;

	{
		// CALL THE LIBRARY THAT DOES THE HEAVYLIFTING
		Recombine(&data);
	}
	// recover the information and resize buffers down to the data
	*pno_kept_locations = data.pOutCloudInfo->No_KeptLocations;
}

void _recombine(size_t stCubatureDegree, ptrdiff_t dimension, ptrdiff_t no_points, std::vector<const void*> vpLocationBuffer, std::vector<double> vdWeightBuffer, std::vector<size_t>& KeptLocations, std::vector<double>& NewWeights)
{
	ptrdiff_t noKeptLocations;
	// get max buffer size
	_recombineC(
		stCubatureDegree
		, dimension
		, 0
		, &noKeptLocations
		// Python extension code built with distutils is compiled with the same set of compiler options,
		// regardless of whether it's C or C++. So we can't have -std=c99 and -std-c++11 simultaneously.
		, NULL // nullptr
		, NULL // nullptr
		, NULL // nullptr
		, NULL // nullptr
	);

	// setup a buffer of size iNoDimensionsToCubature to store indexes to the kept points
	KeptLocations.resize(noKeptLocations);

	// setup a buffer of size iNoDimensionsToCubature to store the weights of the kept points
	NewWeights.resize(noKeptLocations);

	// set up the number of active input points
	ptrdiff_t noLocations = (vpLocationBuffer.size() == vdWeightBuffer.size()) ? vpLocationBuffer.size() : 0;

	_recombineC(
		stCubatureDegree
		, dimension
		, noLocations
		, &noKeptLocations
		, &vpLocationBuffer[0]
		, &vdWeightBuffer[0]
		, &KeptLocations[0]
		, &NewWeights[0]
	);
	// adjust buffer to actual number returned
	NewWeights.resize(noKeptLocations);
	KeptLocations.resize(noKeptLocations);
}
