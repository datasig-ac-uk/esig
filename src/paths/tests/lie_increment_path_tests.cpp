//
// Created by user on 01/04/2022.
//

#include "esig/paths/lie_increment_path.h"

#include <esig/algebra/context.h>
#include <esig/implementation_types.h>

#include <esig/algebra/tensor_interface.h>
#include <esig/algebra/lie_interface.h>
#include <gtest/gtest.h>
#include <random>

namespace esig {
namespace algebra {

inline void PrintTo(const lie& arg, std::ostream* os)
{ arg.print(*os); }

inline void PrintTo(const free_tensor& arg, std::ostream* os)
{ arg.print(*os); }

}
}

#include "RandomScalarsFixture.h"

struct LieIncrementPathTestsFixture : public ::testing::Test, esig::testing::RandomScalars
{
protected:

    static constexpr esig::deg_t width = 5;
    static constexpr esig::deg_t depth = 2;
    std::shared_ptr<const esig::algebra::context> ctx;

    LieIncrementPathTestsFixture()
        : esig::testing::RandomScalars(-2, 2),
          ctx(esig::algebra::get_context(width, depth, esig::scalars::dtl::scalar_type_holder<float>::get_type(), {}))
    {}


protected:
    esig::scalars::owned_scalar_array random_dense_data(esig::dimn_t num_increments)
    {
         return esig::testing::RandomScalars::random_data(ctx->ctype(), num_increments*width);
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

    esig::paths::path_metadata md{
            width,
            depth,
            esig::real_interval(0.0, 1.0),
            ctx,
            ctx->ctype(),
            esig::algebra::vector_type::dense
    };

    auto data = random_dense_data(1);
    esig::algebra::vector_construction_data edata {
        esig::scalars::key_scalar_array(esig::scalars::owned_scalar_array(data)),
        esig::algebra::vector_type::dense
    };
    auto idx = indices(1);
    const esig::paths::lie_increment_path path(std::move(data), idx, md);


    auto lsig = path.log_signature(esig::real_interval(0.0, 1.0), 1, *ctx->get_alike(1));

    auto expected = ctx->construct_lie(edata);

    ASSERT_EQ(lsig, expected);
}
TEST_F(LieIncrementPathTestsFixture, TestLogSignatureTwoIncrementsDepth1)
{

    esig::paths::path_metadata md{
        width,
        depth,
        esig::real_interval(0.0, 1.0),
        ctx,
        ctx->ctype(),
        esig::algebra::vector_type::dense};

    auto data = random_dense_data(2);

    esig::algebra::vector_construction_data edata {
        esig::scalars::key_scalar_array(ctx->ctype()),
        esig::algebra::vector_type::dense
    };
    edata.data.allocate_scalars(width);
    edata.data.type()->convert_copy(edata.data.ptr(), data, width);
    for (int i=0; i<width; ++i) {
        edata.data[i] += data[i + width];
    }


    auto idx = indices(2);
    const esig::paths::lie_increment_path path(std::move(data), std::move(idx), md);

    auto ctx2 = ctx->get_alike(1);

    auto lsig = path.log_signature(esig::real_interval(0.0, 2.0), 1, *ctx2);

    auto expected = ctx->construct_lie(edata);

    ASSERT_EQ(lsig, expected);
}
