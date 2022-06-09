//
// Created by sam on 17/08/2021.
//


#include <gtest/gtest.h>

#include "esig/intervals.h"


using namespace esig;

TEST(dyadic, test_rebase_dyadic_1)
{
    dyadic val{1, 0};
    val.rebase(1);
    ASSERT_TRUE(dyadic_equals(val, dyadic{2, 1}));
}

TEST(dyadic, test_rebase_dyadic_5)
{
    dyadic val{1, 0};
    val.rebase(5);
    ASSERT_TRUE(dyadic_equals(val, dyadic{32, 5}));
}
