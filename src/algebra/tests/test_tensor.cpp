//
// Created by user on 24/03/2022.
//

#include "fixture_base.h"

using namespace esig;

namespace {


struct TensorFixture : public esig::testing::fixture_base
{

private:
    template <typename I>
    esig::algebra::free_tensor make_tensor_impl(const I* start, const I* end)
    {
        esig::algebra::vector_construction_data data(
                start, end, std::get<3>(GetParam())
                );
        return ctx->construct_tensor(data);
    }

public:

    algebra::free_tensor make_sparse_free_tensor(idimn_t size=-1)
    {
        auto make_size = (size == -1) ? ctx->tensor_size(std::get<1>(GetParam())) : dimn_t(size);
        return ctx->construct_tensor(get_construction_data(make_size, algebra::vector_type::sparse));
    }

    algebra::free_tensor make_dense_free_tensor(idimn_t size=-1)
    {
        auto make_size = (size == -1) ? ctx->tensor_size(std::get<1>(GetParam())) : dimn_t(size);
        auto cons_data = get_construction_data(make_size, algebra::vector_type::dense);
        return ctx->construct_tensor(cons_data);
    }

};



}

TEST_P(TensorFixture, TestTensorAdditionDenseData) {
    auto params = GetParam();
    auto tensor_size = ctx->tensor_size(ctx->depth());

    const auto l1 = make_dense_free_tensor();
    const auto l2 = make_dense_free_tensor();
    auto expected = ctx->zero_tensor(std::get<3>(params));
    for (esig::key_type k=0; k<tensor_size; ++k) {
        expected[k] = l1[k] + l2[k];
    }

    auto result = l1.add(l2);
    ASSERT_EQ(expected, result);
}

TEST_P(TensorFixture, TestTensorSubtractionDenseData) {
    auto params = GetParam();
    auto tensor_size = ctx->tensor_size(ctx->depth());

    const auto l1 = make_dense_free_tensor();
    const auto l2 = make_dense_free_tensor();

    auto expected = ctx->zero_tensor(std::get<3>(params));
    for (esig::key_type k=0; k< tensor_size; ++k) {
        expected[k] = l1[k] - l2[k];
    }

    auto result = l1.sub(l2);
    ASSERT_EQ(expected, result);
}

//TEST_P(LieFixture, TestLieSMulDenseData) {
//    auto params = GetParam();
//    auto lie_size = ctx->lie_size(ctx->depth());
//    auto d1 = random_dense_data(lie_size);
//
//    const auto l1 = make_lie_dense_data(d1);
//    double scal(ddist(rng));
//
//    auto expected = ctx->zero_lie(std::get<2>(params), std::get<3>(params));
//    for (esig::key_type k=1; k<=lie_size; ++k) {
//        expected[k] = l1[k]*scal;
//    }
//
//
//}

TEST_P(TensorFixture, TestFMExpInplace) {
    auto tensor_size = ctx->tensor_size(ctx->depth());

    auto l1 = make_dense_free_tensor();
    const auto l2 = make_dense_free_tensor();

    const auto result = l1.fmexp(l2);

}


INSTANTIATE_TEST_SUITE_P(TensorTests, TensorFixture,
                        ::testing::Combine(
                                ::testing::Values(2, 5, 10),
                                ::testing::Values(2, 5),
                                ::testing::Values(esig::scalars::dtl::scalar_type_holder<float>::get_type(), esig::scalars::dtl::scalar_type_holder<double>::get_type()),
                                ::testing::Values(esig::algebra::vector_type::sparse, esig::algebra::vector_type::dense)
                                ),
                        esig::testing::get_param_test_name
                        );
