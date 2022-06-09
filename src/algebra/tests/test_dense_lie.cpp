//
// Created by sam on 11/04/2022.
//
#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/vectors/vectors.h>
#include <libalgebra/coefficients/rational_coefficients.h>

#include "la_esig_equality_helper.h"

#include <random>



using namespace esig;
constexpr deg_t WIDTH = 5;
constexpr deg_t DEPTH = 5;

using la_lie = alg::lie<alg::coefficients::rational_field, WIDTH, DEPTH, alg::vectors::dense_vector>;

class DenseLieTests : public ::testing::Test
{
protected:

    std::random_device rdev;
    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    using basis_t = alg::lie_basis<WIDTH, DEPTH>;
    using scal_t = typename alg::coefficients::rational_field::S;
    using lie = algebra::dense_lie<scal_t>;

    std::shared_ptr<algebra::lie_basis> basis;

    DenseLieTests() : basis(new algebra::lie_basis(WIDTH, DEPTH)), rng(rdev()), dist(-10000, 10000)
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


TEST_F(DenseLieTests, TestUminus)
{
    auto rd = random_data();

    lie t(basis, rd.data(), rd.data() + rd.size());
    la_lie et(rd.data(), rd.data() + rd.size());

    ASSERT_TRUE(LiesEqual(-t, -et));

}




TEST_F(DenseLieTests, TestAdd)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 + t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 + et2;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseLieTests, TestSub)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 - t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 - et2;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseLieTests, TestSMul)
{
    auto rd = random_data();
    lie t1(basis, rd.data(), rd.data() + rd.size());
    auto scal = random_scalar();

    auto result = t1 * scal;

    la_lie et(rd.data(), rd.data() + rd.size());
    auto expected = et * scal;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseLieTests, TestSDiv)
{
    auto rd = random_data();
    lie t1(basis, rd.data(), rd.data() + rd.size());
    auto scal = random_scalar();

    auto result = t1 / scal;

    la_lie et(rd.data(), rd.data() + rd.size());
    auto expected = et / scal;

    ASSERT_TRUE(LiesEqual(result, expected));
}


TEST_F(DenseLieTests, TestMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    auto result = t1 * t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());

    auto expected = et1 * et2;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseLieTests, TestAddInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 += t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());

    et1 += et2;

    ASSERT_TRUE(LiesEqual(t1, et1));
}

TEST_F(DenseLieTests, TestSubInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 -= t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());
    et1 -= et2;

    ASSERT_TRUE(LiesEqual(t1, et1));
}

TEST_F(DenseLieTests, TestSMulInplace)
{
    auto rd1 = random_data();
    auto scal = random_scalar();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());

    t1 *= scal;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    et1 *= scal;

    ASSERT_TRUE(LiesEqual(t1, et1));
}

TEST_F(DenseLieTests, TestSDivInplace)
{
    auto rd1 = random_data();
    auto scal = random_scalar();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());

    t1 /= scal;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    et1 /= scal;

    ASSERT_TRUE(LiesEqual(t1, et1));
}

TEST_F(DenseLieTests, TestMulInplace)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1 *= t2;

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());
    et1 *= et2;

    ASSERT_TRUE(LiesEqual(t1, et1));

}


TEST_F(DenseLieTests, TestMulScalMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_prod(t2, scal);

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_prod(et2, scal);

    ASSERT_TRUE(LiesEqual(t1, et1));
}


TEST_F(DenseLieTests, TestMulScalDiv)
{
    auto rd1 = random_data();
    auto rd2 = random_data();
    auto scal = random_scalar();

    lie t1(basis, rd1.data(), rd1.data() + rd1.size());
    lie t2(basis, rd2.data(), rd2.data() + rd2.size());

    t1.mul_scal_div(t2, scal);

    la_lie et1(rd1.data(), rd1.data() + rd1.size());
    la_lie et2(rd2.data(), rd2.data() + rd2.size());
    et1.mul_scal_div(et2, scal);

    ASSERT_TRUE(LiesEqual(t1, et1));
}
