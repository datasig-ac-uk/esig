//
// Created by user on 24/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_
#define ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_

#include <esig/implementation_types.h>
#include <esig/scalars.h>
#include <esig/algebra/basis.h>
#include <esig/algebra/context.h>

#include <gtest/gtest.h>


#include <random>


namespace esig {
namespace testing {

using param_type = std::tuple<deg_t, deg_t, const scalars::ScalarType *, algebra::VectorType>;

class fixture_base : public ::testing::TestWithParam<param_type>
{
public:
    std::mt19937 rng;
    std::uniform_real_distribution<float> sdist;


    std::shared_ptr<const algebra::context> ctx;

    fixture_base();

    algebra::vector_construction_data get_construction_data(dimn_t size, algebra::VectorType data_type);


};

std::string get_param_test_name(const ::testing::TestParamInfo<param_type>& info);

}
}

#endif//ESIG_PATHS_SRC_ALGEBRA_TESTS_FIXTURE_BASE_H_
