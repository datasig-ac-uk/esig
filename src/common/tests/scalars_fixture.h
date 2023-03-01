//
// Created by user on 21/11/22.
//

#ifndef ESIG_SRC_COMMON_TESTS_SCALARS_FIXTURE_H_
#define ESIG_SRC_COMMON_TESTS_SCALARS_FIXTURE_H_

#include <esig/scalars.h>
#include <gtest/gtest.h>

namespace esig {
namespace testing {

class EsigScalarFixture : public ::testing::Test
{
public:
    using underlying_type = double;
    const scalars::ScalarType * type;

    const scalars::ScalarType * ftype;
    const scalars::ScalarType * dtype;

    EsigScalarFixture()
        : type(scalars::dtl::scalar_type_trait<double>::get_type()),
          dtype(type),
          ftype(scalars::dtl::scalar_type_trait<float>::get_type())
    {}

};


} // namespace testing
} // namespace esig

#endif//ESIG_SRC_COMMON_TESTS_SCALARS_FIXTURE_H_
