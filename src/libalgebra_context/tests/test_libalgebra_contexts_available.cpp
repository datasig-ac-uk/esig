//
// Created by sam on 29/04/22.
//

#include <gtest/gtest.h>
#include <esig/libalgebra_context/libalgebra_context.h>

#include "../src/libalgebra_context_maker.h"

#define NUM_CONTEXTS_IN_DEFAULT 114


TEST(LibalgebraContexts, TestAvailableContexts)
{
    esig::algebra::libalgebra_context_maker maker;

    std::vector<std::tuple<esig::deg_t, esig::deg_t, esig::algebra::coefficient_type>> configs;
    configs.reserve(NUM_CONTEXTS_IN_DEFAULT);
    for (const auto& it : maker.get()) {
        configs.push_back(it.first);
    }

    EXPECT_EQ(configs.size(), NUM_CONTEXTS_IN_DEFAULT);
    for (auto it : configs) {
        EXPECT_TRUE(maker.can_get(std::get<0>(it), std::get<1>(it), std::get<2>(it)));
    }
}
