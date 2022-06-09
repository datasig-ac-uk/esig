//
// Created by user on 24/03/2022.
//

#include "fixture_base.h"



#include <esig/algebra/context.h>
#include <fallback_context.h>

#include <sstream>

namespace esig {
namespace testing {
fixture_base::fixture_base()
    : rng(std::random_device()()), ddist(-2.0, 2.0), sdist(-2.0f, 2.0f), kdist(0, 5)
{
    auto params = GetParam();
    auto width = std::get<0>(params);
    auto depth = std::get<1>(params);

    ctx = algebra::get_context(width, depth, std::get<2>(params));
}

std::vector<char> fixture_base::random_dense_data(dimn_t size)
{
    auto ctype = std::get<2>(GetParam());
    auto itemsize = size_of(ctype);
    std::vector<char> result(size*itemsize);
    switch (ctype) {
        case algebra::coefficient_type::dp_real: {
            assert(itemsize == sizeof(double));
            auto* ptr = reinterpret_cast<double*>(result.data());
            for (auto i=0; i<size; ++i) {
                ptr[i] = ddist(rng);
            }
            }
            break;
        case algebra::coefficient_type::sp_real: {
            assert(itemsize == sizeof(float));
            auto* ptr = reinterpret_cast<float*>(result.data());
            for (auto i=0; i<size; ++i) {
                ptr[i] = sdist(rng);
            }
            }
            break;
    }
    return result;
}
std::vector<char> fixture_base::random_sparse_data(dimn_t size, key_type start, key_type end)
{
    auto ctype = std::get<2>(GetParam());
    std::vector<char> result;
    key_type k = start + kdist(rng);
    switch (ctype) {
        case algebra::coefficient_type::dp_real: {
            result.resize(size*sizeof(std::pair<key_type, double>));
            auto *ptr = reinterpret_cast<std::pair<key_type, double>*>(result.data());
            for (auto i = 0; i < size && k < end; ++i) {
                ptr[i] = std::pair<key_type, double>(k, ddist(rng));
                k += kdist(rng);
            }
        } break;
        case algebra::coefficient_type::sp_real: {
            auto *ptr = reinterpret_cast<std::pair<key_type, float>*>(result.data());
            for (auto i = 0; i < size && k < end; ++i) {
                ptr[i] = std::pair<key_type, float>(k, sdist(rng));
                k += kdist(rng);
            }
        } break;
    }
    return result;
}

std::vector<float> fixture_base::random_fdata(dimn_t size)
{
    std::vector<float> result;
    result.reserve(size);
    for (auto i = 0; i < size; ++i) {
        result.push_back(sdist(rng));
    }
    return result;
}
std::vector<double> fixture_base::random_ddata(dimn_t size)
{
    std::vector<double> result;
    result.reserve(size);
    for (auto i=0; i<size; ++i) {
        result.push_back(ddist(rng));
    }
    return result;
}
std::vector<std::pair<key_type, float>> fixture_base::random_kv_fdata(dimn_t size, key_type start, key_type end_key)
{
    std::vector<std::pair<key_type, float>> result;
    result.reserve(size);
    auto k = start + kdist(rng);
    for (auto i=0; i<size && k < end_key; ++i) {
        result.emplace_back(k, sdist(rng));
        k += kdist(rng);
    }
    return result;
}
std::vector<std::pair<key_type, double>> fixture_base::random_kv_ddata(dimn_t size, key_type start, key_type end_key)
{
    std::vector<std::pair<key_type, double>> result;
    result.reserve(size);
    auto k = start + kdist(rng);
    for (auto i=0; i<size && k < end_key; ++i) {
        result.emplace_back(k, ddist(rng));
        k += kdist(rng);
    }
    return result;
}

std::string get_param_test_name(const ::testing::TestParamInfo<param_type> &info)
{
    std::stringstream ss;
    ss << std::get<0>(info.param) << ""
       << std::get<1>(info.param) << "";
    switch(std::get<2>(info.param)) {
        case algebra::coefficient_type::sp_real: ss << "float"; break;
        case algebra::coefficient_type::dp_real: ss << "double"; break;
    }
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
