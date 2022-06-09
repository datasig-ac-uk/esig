//
// Created by user on 24/03/2022.
//

#include "fixture_base.h"

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


    esig::algebra::lie make_lie_dense_data(const std::vector<char>& data)
    {
        switch (std::get<2>(GetParam())) {
            case esig::algebra::coefficient_type::sp_real: {
                auto* begin = reinterpret_cast<const float*>(data.data());
                auto* end = reinterpret_cast<const float*>(data.data() + data.size());
                return make_lie_impl(begin, end);
            }
            case esig::algebra::coefficient_type::dp_real: {
                auto* begin = reinterpret_cast<const double*>(data.data());
                auto* end = reinterpret_cast<const double*>(data.data() + data.size());
                return make_lie_impl(begin, end);
            }
        }
        throw std::runtime_error("bad data");
    }

    esig::algebra::lie make_lie_sparse_data(const std::vector<char>& data)
    {
        switch(std::get<2>(GetParam())) {
            case esig::algebra::coefficient_type::sp_real: {
                auto* begin = reinterpret_cast<const std::pair<esig::key_type, float>*>(data.data());
                auto* end = reinterpret_cast<const std::pair<esig::key_type, float>*>(data.data()+data.size());
                return make_lie_impl(begin, end);
            }
            case esig::algebra::coefficient_type::dp_real: {
                auto *begin = reinterpret_cast<const std::pair<esig::key_type, double> *>(data.data());
                auto *end = reinterpret_cast<const std::pair<esig::key_type, double> *>(data.data() + data.size());
                return make_lie_impl(begin, end);
            }
        }
        throw std::runtime_error("bad data");
    }

};



}

TEST_P(LieFixture, TestLieAdditionDenseData) {
    auto params = GetParam();
    auto lie_size = ctx->lie_size(ctx->depth());
    auto d1 = random_dense_data(lie_size);
    auto d2 = random_dense_data(lie_size);

    const auto l1 = make_lie_dense_data(d1);
    const auto l2 = make_lie_dense_data(d2);

    auto expected = ctx->zero_lie(std::get<3>(params));
    for (esig::key_type k=1; k<=lie_size; ++k) {
        expected[k] = l1[k] + l2[k];
    }

    auto result = l1.add(l2);
    ASSERT_EQ(expected, result);
}

TEST_P(LieFixture, TestLieSubtractionDenseData) {
    auto params = GetParam();
    auto lie_size = ctx->lie_size(ctx->depth());
    auto d1 = random_dense_data(lie_size);
    auto d2 = random_dense_data(lie_size);

    const auto l1 = make_lie_dense_data(d1);
    const auto l2 = make_lie_dense_data(d2);

    auto expected = ctx->zero_lie(std::get<3>(params));
    for (esig::key_type k=1; k<=lie_size; ++k) {
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


INSTANTIATE_TEST_CASE_P(LieTests, LieFixture,
                        ::testing::Combine(
                                ::testing::Values(2, 5, 10),
                                ::testing::Values(2, 5),
                                ::testing::Values(esig::algebra::coefficient_type::sp_real, esig::algebra::coefficient_type::dp_real),
                                ::testing::Values(esig::algebra::vector_type::sparse, esig::algebra::vector_type::dense)
                                ),
                        esig::testing::get_param_test_name
                        );
