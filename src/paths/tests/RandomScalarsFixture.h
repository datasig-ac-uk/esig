//
// Created by sam on 03/02/23.
//

#ifndef ESIG_SRC_PATHS_TESTS_RANDOMSCALARSFIXTURE_H_
#define ESIG_SRC_PATHS_TESTS_RANDOMSCALARSFIXTURE_H_

#include <esig/implementation_types.h>

#include <random>


#include <esig/scalars.h>




namespace esig {
namespace testing {

class RandomScalars {
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    RandomScalars(float lower, float upper) : dist(lower, upper), rng(std::random_device()())
    {}

    scalars::OwnedScalarArray random_data(const scalars::ScalarType * ctype, std::size_t count);


};

}// namespace testing
}// namespace esig

#endif//ESIG_SRC_PATHS_TESTS_RANDOMSCALARSFIXTURE_H_
