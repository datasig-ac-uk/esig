//
// Created by user on 23/11/22.
//
#include "scalars_fixture.h"

using namespace esig;
using namespace esig::scalars;
using ::esig::testing::EsigScalarFixture;

TEST_F(EsigScalarFixture, default_construct) {
    OwnedScalarArray arry;

    EXPECT_EQ(arry.size(), 0);
    EXPECT_EQ(arry.ptr(), nullptr);
    EXPECT_EQ(arry.type(), nullptr);
}

TEST_F(EsigScalarFixture, construct_with_type) {
    OwnedScalarArray arry(type);

    EXPECT_EQ(arry.size(), 0);
    EXPECT_EQ(arry.ptr(), nullptr);
    EXPECT_EQ(arry.type(), type);
}

TEST_F(EsigScalarFixture, construct_type_and_size) {
    OwnedScalarArray arry(type, 5);

    EXPECT_EQ(arry.size(), 5);
    EXPECT_NE(arry.ptr(), nullptr);
    EXPECT_EQ(arry.type(), type);
}

TEST_F(EsigScalarFixture, construct_value_and_size) {
    Scalar one(1.0, type);
    OwnedScalarArray arry(one, 5);

    EXPECT_EQ(arry.size(), 5);
    EXPECT_NE(arry.ptr(), nullptr);
    EXPECT_EQ(arry.type(), type);

    for (int i=0; i < 5; ++i) {
        EXPECT_EQ(arry[i], one);
    }
}

TEST_F(EsigScalarFixture, from_scalar_array) {
    std::vector<double> base { 1.0, 2.0, 3.0, 4.0, 5.0 };
    ScalarPointer base_begin(base.data(), type);
    ScalarArray original(base_begin, base.size());

    OwnedScalarArray arry(original);

    EXPECT_EQ(arry.size(), base.size());
    EXPECT_NE(arry.ptr(), base.data());
    EXPECT_NE(arry.ptr(), nullptr);
    EXPECT_EQ(arry.type(), type);

    for (auto i=0; i<base.size(); ++i) {
        EXPECT_EQ(arry[i], Scalar(base[i], type));
    }
}
