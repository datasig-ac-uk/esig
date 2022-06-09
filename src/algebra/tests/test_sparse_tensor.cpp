//
// Created by sam on 13/04/2022.
//
#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/vectors/vectors.h>
#include <libalgebra/coefficients/rational_coefficients.h>

#include "la_esig_equality_helper.h"
#include <random>
#include <utility>
#include <vector>


using namespace esig;
constexpr deg_t WIDTH = 5;
constexpr deg_t DEPTH = 5;

using la_tensor = alg::free_tensor<alg::coefficients::rational_field, WIDTH, DEPTH, alg::vectors::sparse_vector>;
//using la_tensor = alg::free_tensor<alg::coefficients::double_field, WIDTH, DEPTH, alg::vectors::sparse_vector>;


class SparseTensorTests : public ::testing::Test
{
protected:

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    using basis_t = alg::free_tensor_basis<WIDTH, DEPTH>;
    using scal_t = typename alg::coefficients::rational_field::S;
//    using scal_t = double;
    using tensor = algebra::sparse_tensor<scal_t>;

    std::shared_ptr<algebra::tensor_basis> basis;

    SparseTensorTests() : basis(new algebra::tensor_basis(WIDTH, DEPTH)), rng(std::random_device{}()), dist(-10000, 10000)
    {}


    std::vector<std::pair<const key_type, scal_t>> random_data()
    {
        std::vector<std::pair<const key_type, scal_t>> result;
        auto sz = basis_t::start_of_degree(DEPTH+1);
        result.reserve(sz);

        dimn_t skip = WIDTH;
        for (auto deg=1; deg <= DEPTH; ++deg) {
            std::uniform_int_distribution<key_type> keydist(1, skip-1);
            key_type key = basis_t::start_of_degree(deg);
            while ((key += keydist(rng)) < basis_t::start_of_degree(deg+1)) {
                result.emplace_back(key, scal_t(dist(rng)) / 10000);
            }

            skip *= WIDTH;
        }

        return result;
    }

    scal_t random_scalar()
    {
        return scal_t(dist(rng)) / 10000;
    }

    template <typename It>
    la_tensor mk_la_tensor(It begin, It end)
    {
        la_tensor result;
        for (auto it = begin; it != end; ++it) {
            result.add_scal_prod(basis_t::index_to_key(it->first), it->second);
        }
        return result;
    }

};




TEST_F(SparseTensorTests, TestUminus)
{
    auto rd = random_data();
    tensor t(basis, rd.begin(), rd.end());
    auto et = mk_la_tensor(rd.begin(), rd.end());

    ASSERT_TRUE(TensorsEqual(-t, -et));
}

TEST_F(SparseTensorTests, TestAdd)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.begin(), rd1.end());
    tensor t2(basis, rd2.begin(), rd2.end());

    auto result = t1 + t2;

    auto et1 = mk_la_tensor(rd1.begin(), rd1.end());
    auto et2 = mk_la_tensor(rd2.begin(), rd2.end());

    auto expected = et1 + et2;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(SparseTensorTests, TestSub)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1 (basis, rd1.begin(), rd1.end());
    tensor t2 (basis, rd2.begin(), rd2.end());

    auto result = t1 - t2;

    auto et1 = mk_la_tensor(rd1.begin(), rd1.end());
    auto et2 = mk_la_tensor(rd2.begin(), rd2.end());

    auto expected = et1 - et2;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(SparseTensorTests, TestSMul)
{
    auto rd = random_data();
    auto scal = random_scalar();

    tensor t(basis, rd.begin(), rd.end());
    auto result = t * scal;

    auto et = mk_la_tensor(rd.begin(), rd.end());
    auto expected = et * scal;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(SparseTensorTests, TestSDiv)
{
    auto rd = random_data();
    auto scal = random_scalar();

    tensor t(basis, rd.begin(), rd.end());
    auto result = t / scal;

    auto et = mk_la_tensor(rd.begin(), rd.end());
    auto expected = et / scal;

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(SparseTensorTests, TestMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    tensor t1(basis, rd1.begin(), rd1.end());
    tensor t2(basis, rd2.begin(), rd2.end());

    auto result = t1 * t2;

    auto et1 = mk_la_tensor(rd1.begin(), rd1.end());
    auto et2 = mk_la_tensor(rd2.begin(), rd2.end());

    auto expected = et1 * et2;

    ASSERT_TRUE(TensorsEqual(result, expected));

}
