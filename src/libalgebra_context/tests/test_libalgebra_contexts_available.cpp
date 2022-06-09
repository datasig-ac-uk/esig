//
// Created by sam on 29/04/22.
//

#include <gtest/gtest.h>
#include <esig/libalgebra_context/libalgebra_context.h>

#include "../src/libalgebra_context_maker.h"

#define NUM_CONTEXTS_IN_DEFAULT 57


TEST(LibalgebraContexts, TestAvailableContexts)
{
    esig::algebra::libalgebra_context_maker maker;

    std::vector<std::pair<esig::deg_t, esig::deg_t>> configs;
    configs.reserve(NUM_CONTEXTS_IN_DEFAULT);
    for (const auto& it : maker.get()) {
        configs.push_back(it.first);
    }

    EXPECT_EQ(configs.size(), NUM_CONTEXTS_IN_DEFAULT);
    for (auto it : configs) {
        EXPECT_TRUE(maker.can_get(it.first, it.second));
    }
}
