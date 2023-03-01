//
// Created by user on 24/03/2022.
//

#include "fixture_base.h"

using namespace esig;

namespace {


struct LieFixture : public esig::testing::fixture_base
{

private:
    template <typename I>
    esig::algebra::lie make_lie_impl(const I* start, const I* end)
    {
        esig::algebra::vector_construction_data data(
                start, end, std::get<3>(GetParam())
                );
        return ctx->construct_lie(data);
    }

public:

    algebra::lie make_sparse_lie(idimn_t size=-1)
    {
        dimn_t make_size = (size == -1) ? ctx->lie_size(std::get<1>(GetParam())) : dimn_t(size);
        return ctx->construct_lie(get_construction_data(make_size, algebra::vector_type::sparse));
    }


    algebra::lie make_dense_lie(idimn_t size=-1)
    {
        dimn_t make_size = (size == -1) ? ctx->lie_size(std::get<1>(GetParam())) : dimn_t(size);
        return ctx->construct_lie(get_construction_data(make_size, algebra::vector_type::dense));
    }


};



}

TEST_P(LieFixture, TestLieAdditionDenseData) {
    auto params = GetParam();
    const auto lie_size = ctx->lie_size(std::get<1>(params));

    const auto l1 = make_dense_lie();
    const auto l2 = make_dense_lie();

    auto expected = ctx->zero_lie(std::get<3>(params));
    for (esig::key_type k=1; k<=lie_size; ++k) {
        auto r = l1[k] + l2[k];
        expected[k] = r;
    }

    auto result = l1.add(l2);
    ASSERT_EQ(expected, result);
}

TEST_P(LieFixture, TestLieSubtractionDenseData) {
    auto params = GetParam();
    auto lie_size = ctx->lie_size(ctx->depth());

    const auto l1 = make_dense_lie();
    const auto l2 = make_dense_lie();

    auto expected = ctx->zero_lie(std::get<3>(params));
    for (esig::key_type k=1; k<=lie_size; ++k) {
        auto r = l1[k] - l2[k];
        expected[k] = r;
    }

    auto result = l1.sub(l2);
    ASSERT_EQ(expected, result) << expected << result;
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


INSTANTIATE_TEST_SUITE_P(LieTests, LieFixture,
                        ::testing::Combine(
                                ::testing::Values(2, 5, 10),
                                ::testing::Values(2, 5),
                                ::testing::Values(scalars::dtl::scalar_type_holder<float>::get_type(), scalars::dtl::scalar_type_holder<double>::get_type()),
                                ::testing::Values(esig::algebra::vector_type::sparse, esig::algebra::vector_type::dense)
                                ),
                        esig::testing::get_param_test_name
                        );
