//
// Created by user on 24/05/22.
//
#include "../src/tick_path.h"

#include <esig/algebra/context.h>
#include <esig/implementation_types.h>
#include <gtest/gtest.h>



struct TickLatticePathFixture : public ::testing::Test
{
protected:

    static constexpr esig::deg_t width = 5;
    static constexpr esig::deg_t depth = 2;
    static constexpr esig::algebra::coefficient_type ctype = esig::algebra::coefficient_type::dp_real;
    std::shared_ptr<esig::algebra::context> ctx;

    esig::paths::path path;

    TickLatticePathFixture()
        : ctx(esig::algebra::get_context(width, depth, ctype, {})),
          path(lattice_path())
    {}


    esig::algebra::lie make_lie(std::vector<std::pair<esig::key_type, double>> data)
    {
        esig::algebra::vector_construction_data cd {
            data.data(), data.data() + data.size(),
            esig::algebra::vector_type::sparse
        };
        return ctx->construct_lie(cd);
    }

private:

    static esig::paths::tick_path lattice_path()
    {
        auto alloc = esig::algebra::allocator_for_key_coeff(ctype);
        esig::algebra::allocating_data_buffer buf(alloc, width);

        assert(alloc->item_size() == sizeof(std::pair<esig::key_type, double>));
        std::vector<std::pair<esig::key_type, double>> data {
                {1, 1.},
                {2, 1.},
                {3, 1.},
                {4, 1.},
                {5, 1.}
        };
        std::copy(data.begin(), data.end(), reinterpret_cast<std::pair<esig::key_type, double>*>(buf.begin()));

        std::vector<std::pair<esig::param_t, esig::dimn_t>> index {
                {1., 1},
                {2., 1},
                {3., 1},
                {4., 1},
                {5., 1}
        };

        esig::paths::path_metadata md {
                width,
                depth,
                esig::real_interval(0., 6.),
                ctype,
                esig::algebra::vector_type::sparse,
                esig::algebra::vector_type::sparse
        };

        return esig::paths::tick_path(std::move(md), std::move(index), std::move(buf));
    }

};


TEST_F(TickLatticePathFixture, testLogSignatureSingleIncrement) {
    auto lsig = path.log_signature(esig::real_interval(0., 1.5), 0.1, *ctx);

    auto expected = make_lie({{1, 1.}});

    std::cout << expected << lsig << '\n';
    ASSERT_EQ(lsig, expected);
}
