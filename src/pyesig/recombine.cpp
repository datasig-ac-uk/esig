//
// Created by user on 09/12/22.
//

#include "recombine.h"


#include <pybind11/numpy.h>
#include <recombine/recombine.h>

using namespace esig;
using namespace pybind11::literals;


static const char* RECOMBINE_DOC = R"edoc(recombine(ensemble, selector=(0,1,2,...no_points-1),
weights = (1,1,..,1), degree = 1) ensemble is a numpy array of vectors of type
NP_DOUBLE referred to as points, the selector is a list of indexes to rows in
the ensemble, weights is a list of positive weights of equal length to the
selector and defines an empirical measure on the points in the ensemble.
Returns (retained_indexes, new_weights). The arrays index_array, weights_array
are single index numpy arrays and must have the same dimension and represent
the indexes of the vectors and a mass distribution of positive weights (and at
least one must be strictly positive) on them. The returned weights are strictly
positive, have the same total mass - but are supported on a subset of the
initial chosen set of locations. If degree has its default value of 1 then the
vector data has the same integral under both weight distributions; if degree is
k then both sets of weights will have the same moments for all degrees at most
k; the indexes returned are a subset of indexes in input index_array and mass
cannot be further recombined onto a proper subset while preserving the integral
and moments. The default is to index of all the points, the default weights are
1.0 on each point indexed. The default degree is one.
)edoc";


static void recombine_wrapper(dimn_t stCubatureDimension,
                              dimn_t dimension,
                              dimn_t no_locations,
                              dimn_t* pno_kept_locations,
                              const void** ppLocationBuffer,
                              scalar_t* pdWeightBuffer,
                              dimn_t* pKeptLocations,
                              scalar_t* pNewWeights)
{
    auto& no_kept_locations = *pno_kept_locations;
    auto iNoDimensionsToCubature = RdToPowersCubatureDimension(dimension, stCubatureDimension);
    if (no_locations == 0) {
        no_kept_locations = iNoDimensionsToCubature;
        return;
    }

    if (no_kept_locations < iNoDimensionsToCubature) {
        no_kept_locations = 0;
        return;
    }

    CMultiDimensionalBufferHelper sConditioning {
        dimension,
        stCubatureDimension
    };

    sCloud in {
        dimn_t(no_locations),
        pdWeightBuffer,
        const_cast<void *>(static_cast<const void *>(ppLocationBuffer)),
        &sConditioning
    };

    sRCloudInfo out {
        iNoDimensionsToCubature,
        pNewWeights,
        pKeptLocations,
        nullptr
    };

    sRecombineInterface data {
        &in,
        &out,
        iNoDimensionsToCubature,
        &RdToPowers,
        nullptr
    };

    Recombine(&data);

    no_kept_locations = data.pOutCloudInfo->No_KeptLocations;
}


static py::tuple py_recombine(const py::array_t<scalar_t>& data,
                       const py::object& src_locations,
                       const py::object& src_weights,
                       std::ptrdiff_t cubature_degree)
{
    if (data.ndim() != 2 || data.shape(0) == 0 || data.shape(1) == 0) {
        throw py::value_error("data is badly formed");
    }

    auto no_data_points = data.shape(0);
    auto point_dimension = data.shape(1);

    py::array_t<std::size_t> src_locs;
    if (!src_locations.is_none()) {
        src_locs = py::array_t<std::size_t>::ensure(src_locations);

        if (src_locs.ndim() != 1 || src_locs.shape(0) == 0) {
            throw py::value_error("source locations badly formed");
        }
    } else {
        src_locs = py::array_t<std::size_t>(py::array::ShapeContainer {data.shape(0)});
        auto* sloc_p = src_locs.mutable_data();
        for (std::size_t i=0;i<data.shape(0); ++i) {
            *(sloc_p++) = i;
        }
    }
    dimn_t no_locations = src_locs.shape(0);

    py::array_t<scalar_t> src_wghts;
    if (!src_weights.is_none()) {
        src_wghts = py::array_t<scalar_t>::ensure(src_weights);

        if (src_wghts.ndim() != 1 || src_wghts.shape(0) == 0) {
            throw py::value_error("source weights badly formed");
        }
    } else {
        src_wghts = py::array_t<scalar_t>(py::array::ShapeContainer {data.shape(0)});
        auto* swht_p = src_wghts.mutable_data();
        for (std::ptrdiff_t i=0; i<data.shape(0); ++i) {
            swht_p[i] = 1.0;
        }
    }

    if (!src_locations.is_none() && !src_weights.is_none() && src_wghts.shape(0) != no_locations) {
        throw py::value_error("source weights and source locations have different sizes");
    }

    if (cubature_degree < 1) {
        throw py::value_error("invalid cubature degree");
    }

    // Normalise the weights
    scalar_t total_mass = 0.0;
    {
        auto *w_ptr = src_wghts.mutable_data();
        for (std::ptrdiff_t i = 0; i < no_locations; ++i) {
            total_mass += w_ptr[i];
        }
        for (std::ptrdiff_t i = 0; i < no_locations; ++i) {
            w_ptr[i] /= total_mass;
        }
    }


    // Get the max number of points needed for the output
    dimn_t no_dimension_to_cubature = 0;
    recombine_wrapper(cubature_degree,
                      point_dimension,
                      0,
                      &no_dimension_to_cubature,
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr);

    auto no_kept_locations = no_dimension_to_cubature;
    py::array_t<dimn_t> kept_locations(py::array::ShapeContainer {no_kept_locations});
    py::array_t<scalar_t> new_weights(py::array::ShapeContainer {no_kept_locations});

    {
        std::vector<const void*> locations_map;
        locations_map.reserve(no_locations);
        auto locs = src_locs.unchecked();
        for (std::ptrdiff_t i=0; i<no_locations; ++i) {
            locations_map.push_back(
                reinterpret_cast<const void*>(data.data(locs[i], 0))
            );
        }

//        _recombineC(cubature_degree,
//                    point_dimension,
//                    no_locations,
//                    &no_kept_locations,
//                    locations_map.data(),
//                    src_wghts.mutable_data(),
//                    reinterpret_cast<std::size_t*>(kept_locations.mutable_data()),
//                    new_weights.mutable_data()
//            );
        recombine_wrapper(cubature_degree,
                          point_dimension,
                          no_locations,
                          &no_kept_locations,
                          locations_map.data(),
                          src_wghts.mutable_data(),
                          static_cast<dimn_t*>(kept_locations.mutable_data()),
                          new_weights.mutable_data());

        kept_locations.resize(py::array::ShapeContainer {no_kept_locations});
        new_weights.resize(py::array::ShapeContainer {no_kept_locations});
    }

    // Un-normalise the weights
    {
        auto* w_ptr = new_weights.mutable_data();
        for (std::ptrdiff_t i=0; i<no_kept_locations; ++i) {
            w_ptr[i] *= total_mass;
        }
    }

    return py::make_tuple(kept_locations, new_weights);
}



void python::init_recombine(py::module_ &m) {

    m.def("recombine",
          &py_recombine,
          "data"_a,
          "src_locations"_a,
          "src_weights"_a,
          "cubature_degree"_a,
          RECOMBINE_DOC);

}
