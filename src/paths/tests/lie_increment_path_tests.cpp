//
// Created by user on 01/04/2022.
//

#include "../src/lie_increment_path.h"
#include <esig/algebra/context.h>
#include <esig/implementation_types.h>
#include <gtest/gtest.h>
#include <random>


struct LieIncrementPathTestsFixture : public ::testing::Test
{
protected:

    static constexpr esig::deg_t width = 5;
    static constexpr esig::deg_t depth = 2;
    std::shared_ptr<const esig::algebra::context> ctx;

    LieIncrementPathTestsFixture()
        : ctx(esig::algebra::get_context(width, depth, esig::algebra::coefficient_type::sp_real, {}))
    {}

private:
    template <typename S>
    esig::algebra::rowed_data_buffer dense_data(esig::deg_t num_increments) const
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<S> dist(-2, 2);

        esig::algebra::rowed_data_buffer data(ctx->coefficient_alloc(), width, num_increments);
        auto* ptr = reinterpret_cast<S*>(data.begin());

        for (auto i = 0; i<num_increments; ++i) {
            for (auto j=0; j<width; ++j) {
                *(ptr++) = dist(rng);
            }
        }
        return data;
    }

protected:
     esig::algebra::rowed_data_buffer random_dense_data(
            esig::deg_t num_increments,
            esig::algebra::coefficient_type ctype = esig::algebra::coefficient_type::sp_real
            ) const
    {

        switch (ctype ) {
            case esig::algebra::coefficient_type::dp_real:
                return dense_data<double>(num_increments);
            case esig::algebra::coefficient_type::sp_real:
                return dense_data<float>(num_increments);
        }
        throw std::runtime_error("bad");
    }

    std::vector<esig::param_t> indices(esig::deg_t num_increments) const
    {
        if (num_increments == 1) {
            return {0.0};
        }

        std::vector<esig::param_t> result;
        result.reserve(num_increments);

        esig::param_t step(1.0 / (num_increments - 1));
        for (auto i=0; i<num_increments; ++i) {
            result.push_back(i*step);
        }
        return result;
    }
};


TEST_F(LieIncrementPathTestsFixture, TestLogSignatureSingleIncrement)
{
    auto data = random_dense_data(1);
    auto idx = indices(1);

    esig::paths::path_metadata md{
            width,
            depth,
            esig::real_interval(0.0, 1.0),
            esig::algebra::get_context(width, depth, esig::algebra::coefficient_type::sp_real),
            esig::algebra::coefficient_type::sp_real,
            esig::algebra::input_data_type::value_array,
            esig::algebra::vector_type::dense
    };

    esig::paths::lie_increment_path path(esig::algebra::rowed_data_buffer(data), std::move(idx), md);


    auto lsig = path.log_signature(esig::real_interval(0.0, 1.0), 1, *ctx);

    esig::algebra::vector_construction_data edata(
            reinterpret_cast<const float*>(data.begin()),
            reinterpret_cast<const float*>(data.end()),
            esig::algebra::vector_type::dense
            );

    auto expected = ctx->construct_lie(edata);

    ASSERT_EQ(lsig, expected);
}
TEST_F(LieIncrementPathTestsFixture, TestLogSignatureTwoIncrementsDepth1)
{
    auto data = random_dense_data(2);
    auto idx = indices(2);

    esig::paths::path_metadata md{
            width,
            depth,
            esig::real_interval(0.0, 2.0),
            esig::algebra::get_context(width, depth, esig::algebra::coefficient_type::sp_real),
            esig::algebra::coefficient_type::sp_real,
            esig::algebra::input_data_type::value_array,
            esig::algebra::vector_type::dense
    };

    esig::paths::lie_increment_path path(esig::algebra::rowed_data_buffer(data), std::move(idx), md);

    auto ctx2 = ctx->get_alike(1);

    auto lsig = path.log_signature(esig::real_interval(0.0, 2.0), 1, *ctx2);

    std::vector<float> new_data;
    new_data.reserve(width);
    auto* ptr = reinterpret_cast<const float*>(data.begin());
    for (int i=0; i<width; ++i) {
        new_data.push_back(ptr[i] + ptr[i+width]);
    }

    esig::algebra::vector_construction_data edata(
            reinterpret_cast<const float*>(new_data.data()),
            reinterpret_cast<const float*>(new_data.data() + new_data.size()),
            esig::algebra::vector_type::dense
            );

    auto expected = ctx->construct_lie(edata);

    ASSERT_EQ(lsig, expected);
}
