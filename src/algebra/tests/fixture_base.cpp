//
// Created by user on 24/03/2022.
//

#include "fixture_base.h"



#include <esig/algebra/context.h>


#include <sstream>

namespace esig {
namespace testing {
fixture_base::fixture_base()
    : rng(std::random_device()()), sdist(-2.0f, 2.0f)
{
    auto params = GetParam();
    auto width = std::get<0>(params);
    auto depth = std::get<1>(params);

    ctx = algebra::get_context(width, depth, std::get<2>(params));
}

algebra::vector_construction_data fixture_base::get_construction_data(dimn_t size, algebra::vector_type data_type) {
    std::vector<float> tmp_data;
    tmp_data.reserve(size);

    scalars::key_scalar_array ksa(std::get<2>(GetParam()));
    if (data_type == algebra::vector_type::sparse) {
        ksa.allocate_keys(idimn_t(size));
        auto* keys = ksa.keys();

        for (key_type i=0; i<size; ++i, ++keys) {
            ::new (keys) key_type(i);
        }
    }

    for (dimn_t i=0; i<size; ++i) {
        tmp_data.push_back(sdist(rng));
    }

    ksa.allocate_scalars(idimn_t(size));
    scalars::scalar_pointer in_ptr(tmp_data.data());
    ksa.type()->convert_copy(ksa.ptr(), in_ptr, size);

    return {std::move(ksa), std::get<3>(GetParam())};
}

std::string get_param_test_name(const ::testing::TestParamInfo<param_type> &info)
{
    std::stringstream ss;
    ss << std::get<0>(info.param) << ""
       << std::get<1>(info.param) << ""
       << std::get<2>(info.param)->info().name << "";
    ss << "";
    switch(std::get<3>(info.param)) {
        case algebra::vector_type::sparse:
            ss << "sparse";
            break;
        case algebra::vector_type::dense:
            ss << "dense";
            break;
    }
    return ss.str();
}


}
}
