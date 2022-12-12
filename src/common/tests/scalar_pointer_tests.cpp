//
// Created by user on 21/11/22.
//


#include "scalars_fixture.h"


using namespace esig;

using ::esig::testing::EsigScalarFixture;

TEST_F(EsigScalarFixture, construct_const_pointer)
{
    const underlying_type s(0);
    scalar_pointer ptr(&s, type);

    EXPECT_FALSE(ptr.is_null());
    EXPECT_TRUE(ptr.is_const());
}

TEST_F(EsigScalarFixture, construct_mut_pointer)
{
    underlying_type s(0);
    scalar_pointer ptr(&s, type);

    EXPECT_FALSE(ptr.is_null());
    EXPECT_FALSE(ptr.is_const());
}

TEST_F(EsigScalarFixture, scalar_pointer_offset)
{
    underlying_type array[5] {};
    scalar_pointer ptr(array, type);

    for (dimn_t i=0; i<5; ++i) {
        auto new_p = ptr + i;
        EXPECT_EQ(array + i, new_p.ptr());
    }
}

TEST_F(EsigScalarFixture, scalar_pointer_equals)
{
    underlying_type one(1), two(2);
    scalar_pointer p1(&one, type), p2(&one, type), p3(&two, type);

    EXPECT_EQ(p1, p2);
    EXPECT_NE(p1, p3);
}

TEST_F(EsigScalarFixture, const_pointer_mut_ptr_fail) {
    const underlying_type s(1);
    scalar_pointer ptr(&s, type);

    EXPECT_NO_THROW(ptr.deref());
    EXPECT_THROW(ptr.deref_mut(), std::runtime_error);
}
