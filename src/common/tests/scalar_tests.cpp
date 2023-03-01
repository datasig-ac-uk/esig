//
// Created by user on 21/11/22.
//

#include "scalars_fixture.h"

using namespace esig;
using namespace esig::scalars;
using esig::testing::EsigScalarFixture;

TEST_F(EsigScalarFixture, test_creation_default)
{
    Scalar s;

    EXPECT_TRUE(s.is_zero());
    EXPECT_TRUE(s.is_value());
    EXPECT_FALSE(s.is_const());
}

TEST_F(EsigScalarFixture, test_creation_with_scalar_t)
{
    static_assert(std::is_same<underlying_type, scalar_t>::value,
                  "This test does not make sense with a non-double underlying type"
                  );
    Scalar s(1.0);

    EXPECT_EQ(s.type(), type);
    EXPECT_TRUE(s.is_value());
    EXPECT_FALSE(s.is_const());
}

TEST_F(EsigScalarFixture, to_const_pointer_roundtrip)
{
    underlying_type s(1);
    ScalarPointer p(&s, type);
    Scalar scal(p);

    EXPECT_EQ(scal.to_const_pointer(), p);
}

TEST_F(EsigScalarFixture, to_pointer_roundtrip)
{
    underlying_type s(1);
    ScalarPointer p(&s, type);
    Scalar scal(p);

    EXPECT_EQ(scal.to_pointer(), p);
}

TEST_F(EsigScalarFixture, to_mut_pointer_const_fail)
{
    const underlying_type s(1);
    ScalarPointer p(&s, type);
    Scalar scal(p);

    EXPECT_TRUE(scal.is_const());
    EXPECT_THROW(scal.to_pointer(), std::runtime_error);
}



TEST_F(EsigScalarFixture, minus)
{
    Scalar begin(underlying_type(1), type);
    Scalar expected(underlying_type(-1), type);

    auto result = -begin;

    EXPECT_EQ(result, expected);
}


TEST_F(EsigScalarFixture, addition_like_scalars)
{
    Scalar lhs(underlying_type(1), type), rhs(underlying_type(2), type);

    auto result = lhs + rhs;
    Scalar expected(underlying_type(3), type);

    EXPECT_EQ(result, expected);
}

TEST_F(EsigScalarFixture, subtraction_like_scalars)
{
    Scalar lhs(underlying_type(2), type), rhs(underlying_type(1), type);
    auto result = lhs - rhs;
    Scalar expected(underlying_type(1), type);

    EXPECT_EQ(result, expected);
}

TEST_F(EsigScalarFixture, multiplication_like_scalars)
{
    Scalar lhs(underlying_type(2), type), rhs(underlying_type(3), type);
    auto result = lhs * rhs;
    Scalar expected(underlying_type(6), type);

    EXPECT_EQ(result, expected);
}

TEST_F(EsigScalarFixture, division_like_scalars)
{
    Scalar lhs(underlying_type(6), type), rhs(underlying_type(3), type);
    auto result = lhs / rhs;
    Scalar expected(underlying_type(2), type);

    EXPECT_EQ(result, expected);
}

TEST_F(EsigScalarFixture, addition_inplace_like_scalars)
{
    Scalar lhs(underlying_type(1), type), rhs(underlying_type(2), type),
        expected(underlying_type(3), type);

    lhs += rhs;

    EXPECT_EQ(lhs, expected);
}
TEST_F(EsigScalarFixture, subtraction_inplace_like_scalars)
{
    Scalar lhs(underlying_type(1), type), rhs(underlying_type(2), type),
        expected(underlying_type(-1), type);

    lhs -= rhs;

    EXPECT_EQ(lhs, expected);
}
TEST_F(EsigScalarFixture, multiplication_inplace_like_scalars)
{
    Scalar lhs(underlying_type(2), type), rhs(underlying_type(3), type),
        expected(underlying_type(6), type);

    lhs *= rhs;

    EXPECT_EQ(lhs, expected);
}
TEST_F(EsigScalarFixture, division_inplace_like_scalars)
{
    Scalar lhs(underlying_type(6), type), rhs(underlying_type(2), type),
        expected(underlying_type(3), type);

    lhs /= rhs;

    EXPECT_EQ(lhs, expected);
}

TEST_F(EsigScalarFixture, addition_inplace_const_fails)
{
    const underlying_type s(1);
    Scalar lhs(ScalarPointer{&s, type}), rhs(underlying_type(2), type);

    EXPECT_TRUE(lhs.is_const());
    EXPECT_THROW(lhs += rhs, std::runtime_error);
}

TEST_F(EsigScalarFixture, subtraction_inplace_const_fails)
{
    const underlying_type s(1);
    Scalar lhs(ScalarPointer{&s, type}), rhs(underlying_type(2), type);

    EXPECT_TRUE(lhs.is_const());
    EXPECT_THROW(lhs -= rhs, std::runtime_error);
}

TEST_F(EsigScalarFixture, multiply_inplace_const_fails)
{
    const underlying_type s(1);
    Scalar lhs(ScalarPointer{&s, type}), rhs(underlying_type(2), type);

    EXPECT_TRUE(lhs.is_const());
    EXPECT_THROW(lhs *= rhs, std::runtime_error);
}

TEST_F(EsigScalarFixture, division_inplace_const_fails)
{
    const underlying_type s(1);
    Scalar lhs(ScalarPointer{&s, type}), rhs(underlying_type(2), type);

    EXPECT_TRUE(lhs.is_const());
    EXPECT_THROW(lhs /= rhs, std::runtime_error);
}

TEST_F(EsigScalarFixture, addition_with_default)
{
    Scalar s(underlying_type(1), type);

    auto result = s + Scalar();

    EXPECT_EQ(result, s);
}

TEST_F(EsigScalarFixture, subtract_with_default)
{
    Scalar s(underlying_type(1), type);

    auto result = s - Scalar();

    EXPECT_EQ(result, s);
}

TEST_F(EsigScalarFixture, multiply_with_default)
{
    Scalar s(underlying_type(1), type);

    auto result = s * Scalar();

    EXPECT_TRUE(result.is_zero());
    EXPECT_EQ(result, Scalar(underlying_type(0), type));
}

TEST_F(EsigScalarFixture, division_with_default_fails)
{
    Scalar s(underlying_type(1), type);

    EXPECT_THROW(s / Scalar(), std::runtime_error);
}

TEST_F(EsigScalarFixture, addition_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    auto result = lhs + rhs;

    EXPECT_EQ(result, rhs);
}

TEST_F(EsigScalarFixture, subtraction_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    auto result = lhs + rhs;

    EXPECT_EQ(result, rhs);
}

TEST_F(EsigScalarFixture, multiplication_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    auto result = lhs * rhs;

    EXPECT_EQ(result, type->zero());
}

TEST_F(EsigScalarFixture, division_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    auto result = lhs / rhs;

    EXPECT_EQ(result, type->zero());
}

TEST_F(EsigScalarFixture, addition_default_with)
{
    Scalar lhs, rhs(1.0, type);

    auto result = lhs + rhs;

    EXPECT_EQ(result, rhs);
}

TEST_F(EsigScalarFixture, subtraction_default_with)
{
    Scalar lhs, rhs(1.0, type);

    auto result = lhs + rhs;

    EXPECT_EQ(result, rhs);
}

TEST_F(EsigScalarFixture, multiplication_default_with)
{
    Scalar lhs, rhs(1.0, type);

    auto result = lhs * rhs;

    EXPECT_EQ(result, type->zero());
}

TEST_F(EsigScalarFixture, division_default_with)
{
    Scalar lhs, rhs(1.0, type);

    auto result = lhs / rhs;

    EXPECT_EQ(result, type->zero());
}

TEST_F(EsigScalarFixture, inplace_addition_with_default)
{
    Scalar lhs(1.0, type), rhs;

    lhs += rhs;

    EXPECT_EQ(lhs, Scalar(1.0, type));
}

TEST_F(EsigScalarFixture, inplace_subtraction_with_default)
{
    Scalar lhs(1.0, type), rhs;

    lhs -= rhs;

    EXPECT_EQ(lhs, Scalar(1.0, type));
}

TEST_F(EsigScalarFixture, inplace_multiply_with_default)
{
    Scalar lhs(1.0, type), rhs;

    lhs *= rhs;

    EXPECT_EQ(lhs, Scalar(0.0, type));
}

TEST_F(EsigScalarFixture, inplace_division_with_default)
{
    Scalar lhs(1.0, type), rhs;

    EXPECT_THROW(lhs /= rhs, std::runtime_error);
}

TEST_F(EsigScalarFixture, inplace_addition_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    lhs += rhs;

    EXPECT_EQ(lhs, Scalar(1.0, type));
}

TEST_F(EsigScalarFixture, inplace_subtraction_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    lhs -= rhs;

    EXPECT_EQ(lhs, Scalar(-1.0, type));
}

TEST_F(EsigScalarFixture, inplace_multiply_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    lhs *= rhs;

    EXPECT_EQ(lhs, Scalar(0.0, type));
}

TEST_F(EsigScalarFixture, inplace_division_default_with_type_with)
{
    Scalar lhs(type), rhs(1.0, type);

    lhs /= rhs;

    EXPECT_EQ(lhs, Scalar(0.0, type));
}

TEST_F(EsigScalarFixture, inplace_addition_default_with)
{
    Scalar lhs, rhs(1.0, type);

    lhs += rhs;

    EXPECT_EQ(lhs, Scalar(1.0, type));
}


TEST_F(EsigScalarFixture, inplace_subtraction_default_with)
{
    Scalar lhs, rhs(1.0, type);

    lhs -= rhs;

    EXPECT_EQ(lhs, Scalar(-1.0, type));
}


TEST_F(EsigScalarFixture, inplace_multiply_default_with)
{
    Scalar lhs, rhs(1.0, type);

    lhs *= rhs;

    EXPECT_EQ(lhs, Scalar(0.0, type));
}

TEST_F(EsigScalarFixture, inplace_divide_default_with)
{
    Scalar lhs, rhs(1.0, type);

    lhs /= rhs;

    EXPECT_EQ(lhs, Scalar(0.0, type));
}




/*
 * From here on out, the tests involve type conversions. For these tests,
 * we're going to be converting between floats and doubles. This won't capture
 * some of the weirdness that can occur since we can convert between these two
 * types without any real trouble.
 */

TEST_F(EsigScalarFixture, double_construction_with_float)
{
    Scalar arg(3.14152f, dtype);

    EXPECT_EQ(arg.type(), dtype);
    EXPECT_NEAR(*(const double*) arg.to_const_pointer().ptr(), 3.14152, 2e-7);
}

TEST_F(EsigScalarFixture, float_construction_with_scalar_t)
{
    double val = 3.1525216432134267321;
    float truncated(val);
    Scalar arg(val, ftype);

    EXPECT_EQ(arg.type(), ftype);
    EXPECT_EQ(arg, Scalar(truncated, ftype));
}
