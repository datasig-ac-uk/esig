//
// Created by user on 24/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_
#define ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_

#include <esig/implementation_types.h>
#include <esig/algebra/basis.h>
#include <esig/algebra/context.h>

#include <lie_basis/lie_basis.h>
#include <tensor_basis/tensor_basis.h>

#include <gtest/gtest.h>


#include <random>


namespace esig {
namespace testing {

using param_type = std::tuple<deg_t, deg_t, algebra::coefficient_type, algebra::vector_type>;

class fixture_base : public ::testing::TestWithParam<param_type>
{
public:
    std::mt19937 rng;
    std::uniform_real_distribution<double> ddist;
    std::uniform_real_distribution<float> sdist;
    std::uniform_int_distribution<key_type> kdist;


    std::shared_ptr<algebra::context> ctx;

    fixture_base();

    std::vector<char> random_dense_data(dimn_t size);
    std::vector<char> random_sparse_data(dimn_t size, key_type start, key_type end);

    std::vector<float> random_fdata(dimn_t size);
    std::vector<double> random_ddata(dimn_t size);
    std::vector<std::pair<key_type, float>> random_kv_fdata(dimn_t size, key_type start, key_type end_key);
    std::vector<std::pair<key_type, double>> random_kv_ddata(dimn_t size, key_type start, key_type end_key);
};

std::string get_param_test_name(const ::testing::TestParamInfo<param_type>& info);

}
}

#endif//ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_
