//
// Created by user on 05/04/2022.
//

#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/vectors.h>
#include <libalgebra/rational_coefficients.h>
#include "../src/dense_tensor.h"

#include "la_esig_equality_helper.h"


#include <gtest/gtest.h>

#include <random>

using namespace esig;


constexpr deg_t WIDTH = 2;
constexpr deg_t DEPTH = 5;

using la_tensor = alg::free_tensor<alg::coefficients::rational_field, WIDTH, DEPTH, alg::vectors::dense_vector>;


class DenseTensorTests : public ::testing::Test
{
protected:

    std::random_device rdev;

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    using basis_t = alg::free_tensor_basis<WIDTH, DEPTH>;
    using scal_t = typename alg::coefficients::rational_field::S;
    using tensor = algebra::dense_tensor<scal_t>;

    std::shared_ptr<algebra::tensor_basis> basis;

    DenseTensorTests() : basis(new algebra::tensor_basis(WIDTH, DEPTH)), rng(rdev()), dist(-10000, 10000)
    {}

    std::vector<scal_t> random_data()
    {
        std::vector<scal_t> result;
        auto sz = basis_t::start_of_degree(DEPTH+1);
        result.reserve(sz);

        for (auto i=0; i<sz; ++i) {
            result.push_back(scal_t(dist(rng)) / 10000);
        }

        return result;
    }

    scal_t random_scalar()
    {
        return scal_t(dist(rng)) / 10000;
    }

};


TEST_F(DenseTensorTests, TestUminus)
{
    auto rd = random_data();

    tensor t(basis, rd.data(), rd.data() + rd.size());
    la_tensor et(rd.data(), rd.data() + rd.size());

    ASSERT_TRUE(TensorsEqual(-t, -et));

}




TEST_F(DenseTensorTests, TestAdd)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 + t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 + et2;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestSub)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 - t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 - et2;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestSMul)
{
    auto rd = random_data();
    tensor t1(basis, rd.data(), rd.data() + rd.size());
    auto scal = random_scalar();

    auto result = t1 * scal;

    la_tensor et(rd.data(), rd.data() + rd.size());
    auto expected = et * scal;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestSDiv)
{
    auto rd = random_data();
    tensor t1(basis, rd.data(), rd.data() + rd.size());
    auto scal = random_scalar();

    auto result = t1 / scal;

    la_tensor et(rd.data(), rd.data() + rd.size());
    auto expected = et / scal;

    ASSERT_TRUE(TensorsEqual(result, expected));
}


TEST_F(DenseTensorTests, TestMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 * t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 * et2;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestAddInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 += t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());

    et1 += et2;

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestSubInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 -= t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1 -= et2;

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestSMulInplace)
{
    auto rd1 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());

    t1 *= scal;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    et1 *= scal;

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestSDivInplace)
{
    auto rd1 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());

    t1 /= scal;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    et1 /= scal;

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestMulInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 *= t2;

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1 *= et2;

    ASSERT_TRUE(TensorsEqual(t1, et1));

}


TEST_F(DenseTensorTests, TestMulScalMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_prod(t2, scal);

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_prod(et2, scal);

    ASSERT_TRUE(TensorsEqual(t1, et1));
}


TEST_F(DenseTensorTests, TestMulScalDiv)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_div(t2, scal);

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_div(et2, scal);

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestMulScalMulMax)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_prod(t2, scal, 3);

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_prod(et2, scal, 3);

    ASSERT_TRUE(TensorsEqual(t1, et1));
}



TEST_F(DenseTensorTests, TestMulScalDivMax)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_div(t2, scal, 3);

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_div(et2, scal, 3);

    ASSERT_TRUE(TensorsEqual(t1, et1));
}




TEST_F(DenseTensorTests, TestExp)
{
    auto rd = random_data();
    rd[0] = scal_t(0);

    tensor t(basis, rd.data(), rd.data() + rd.size());
    auto result = exp(t);

    la_tensor et(rd.data(), rd.data() + rd.size());
    auto expected = exp(et);

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestLog)
{
    auto rd = random_data();

    tensor t(basis, rd.data(), rd.data() + rd.size());
    auto result = log(t);

    la_tensor et(rd.data(), rd.data() + rd.size());
    auto expected = log(et);

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseTensorTests, TestFMExp)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());
    t1.fmexp_inplace(t2);

    la_tensor et1(rd1.data(), rd1.data() + rd1.size());
    la_tensor et2(rd2.data(), rd2.data() + rd2.size());
    et1.fmexp_inplace(et2);

    ASSERT_TRUE(TensorsEqual(t1, et1));
}

TEST_F(DenseTensorTests, TestFMExpVsExp) {
    auto rd1 = random_data();
    auto rd2 = random_data();
    rd2[0] = scal_t(0);

    tensor t1(basis, rd1.data(), rd1.data() + rd1.size());
    tensor t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto et1 = t1 * exp(t2);
    t1.fmexp_inplace(t2);

    ASSERT_EQ(t1, et1);
}
