//
// Created by sam on 04/07/22.
//


#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/vectors/vectors.h>
#include <libalgebra/coefficients/rational_coefficients.h>

#include "fallback_context.h"



#include <memory>
#include <random>
#include <vector>
#include "la_esig_equality_helper.h"
#include <gtest/gtest.h>

using namespace esig;
constexpr deg_t WIDTH = 5;
constexpr deg_t DEPTH = 5;


class MapsTests : public ::testing::Test
{
protected:

    std::random_device rdev;
    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    using coeff_t = alg::coefficients::rational_field;
    using scal_t = typename alg::coefficients::rational_field::S;

    MapsTests() : rdev(), rng(rdev()), dist(-10000, 10000)
    {}

    std::vector<scal_t> random_dense_data(std::size_t n);
    std::vector<std::pair<key_type, scal_t>> random_sparse_data(key_type max_key);
    scal_t random_scalar();


};


class DenseMapsTests : public MapsTests
{
protected:

    using lie = algebra::dense_lie<scal_t>;
    using tensor = algebra::dense_tensor<scal_t>;
    using la_lie = alg::lie<coeff_t, WIDTH, DEPTH, alg::vectors::dense_vector>;
    using la_tensor = alg::free_tensor<coeff_t, WIDTH, DEPTH, alg::vectors::dense_vector>;

    algebra::fallback_context ctx;
    std::shared_ptr<algebra::lie_basis> lie_basis;
    std::shared_ptr<algebra::tensor_basis> tensor_basis;

    lie lie_1, lie_2;
    la_lie la_lie_1, la_lie_2;
    tensor tensor_1, tensor_2;
    la_tensor la_tensor_1, la_tensor_2;
    alg::maps<coeff_t, WIDTH, DEPTH, la_tensor, la_lie> la_maps;
    alg::cbh<coeff_t, WIDTH, DEPTH, la_tensor, la_lie> la_cbh;
    esig::algebra::maps maps;


    DenseMapsTests() : ctx(WIDTH, DEPTH, algebra::coefficient_type::dp_real),
                       lie_basis(ctx.get_lie_basis()),
                       tensor_basis(ctx.get_tensor_basis()),
                       lie_1(lie_basis),
                       lie_2(lie_basis),
                       tensor_1(tensor_basis),
                       tensor_2(tensor_basis),
                       maps(tensor_basis, lie_basis)
    {
        const auto& la_lie_basis = la_lie::basis;
        auto lie_size = la_lie_basis.start_of_degree(DEPTH+1);
        auto lie_data_1 = random_dense_data(lie_size);
        auto lie_data_2 = random_dense_data(lie_size);

        lie_1 = lie(lie_basis, lie_data_1.data(), lie_data_1.data() + lie_size);
        lie_2 = lie(lie_basis, lie_data_2.data(), lie_data_2.data() + lie_size);
        la_lie_1 = la_lie(lie_data_1.data(), lie_data_1.data() + lie_size);
        la_lie_2 = la_lie(lie_data_2.data(), lie_data_2.data() + lie_size);

        const auto &la_tensor_basis = la_tensor::basis;
        auto tensor_size = la_tensor_basis.start_of_degree(DEPTH + 1);
        auto tensor_data_1 = random_dense_data(tensor_size);
        auto tensor_data_2 = random_dense_data(tensor_size);

        tensor_1 = tensor(tensor_basis, tensor_data_1.data(), tensor_data_1.data() + tensor_size);
        tensor_2 = tensor(tensor_basis, tensor_data_2.data(), tensor_data_2.data() + tensor_size);
        la_tensor_1 = la_tensor(tensor_data_1.data(), tensor_data_1.data() + tensor_size);
        la_tensor_2 = la_tensor(tensor_data_2.data(), tensor_data_2.data() + tensor_size);
    }

};


TEST_F(DenseMapsTests, TestLieToTensor) {

    auto result = maps.lie_to_tensor<lie, tensor>(lie_1);
    auto expected = la_maps.l2t(la_lie_1);

    ASSERT_TRUE(TensorsEqual(result, expected));
}

TEST_F(DenseMapsTests, TestTensorToLie) {
    tensor new_tensor(tensor_1);
    new_tensor[0] = scal_t(0);
    la_tensor new_la_tensor(la_tensor_1);
    new_la_tensor[typename la_tensor::KEY()] = scal_t(0);

    auto result = maps.tensor_to_lie<lie, tensor>(new_tensor);
    auto expected = la_maps.t2l(new_la_tensor);

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseMapsTests, TestCBH) {
    std::vector<lie> lies {lie_1, lie_2};
    std::vector<la_lie> la_lies {la_lie_1, la_lie_2};

    auto result = maps.cbh(lies.begin(), lies.end());
    auto expected = la_cbh.full(la_lies.begin(), la_lies.end());

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(DenseMapsTests, TestBracketing) {
    const auto& tbasis = la_tensor::basis;
    key_type k = 0;
    for (auto tkey : tbasis.iterate_keys_from(typename la_tensor::KEY(alg::LET(1)))) {
        ++k;
        std::cout << "k " << k << " tkey " << tkey << '\n';

        la_lie expected = la_maps.rbraketing(tkey);
        std::cout << "expected " << expected << '\n';
        la_lie result;
        for (auto r : maps.rbracket(k)) {
            std::cout << r.first << ' ' << r.second << '\n';
            result.add_scal_prod(r.first, scal_t(r.second));
            std::cout << "result " << result << '\n';
        }

        EXPECT_EQ(result, expected);
    }



}



















std::vector<alg::coefficients::rational_field::S> MapsTests::random_dense_data(std::size_t n) {
    std::vector<scal_t> result;
    result.reserve(n);
    for (auto i=0; i<n; ++i) {
        result.push_back(random_scalar());
    }
    return result;
}
std::vector<std::pair<key_type, alg::coefficients::rational_field::S>> MapsTests::random_sparse_data(key_type max_key) {
    return {};
}
alg::coefficients::rational_field::S MapsTests::random_scalar() {
    return scal_t(dist(rng)) / 10000;
}
