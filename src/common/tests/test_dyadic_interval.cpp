#pragma ide diagnostic ignored "cert-err58-cpp"
//
// Created by sam on 17/08/2021.
//

#include <iostream>

#include "esig/intervals.h"
#include <gtest/gtest.h>

using namespace esig;


TEST(dyadic_intervals, test_dincluded_end_clopen)
{
    dyadic_interval di;
    dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dincluded_end(), expected));
}

TEST(dyadic_intervals, test_dexcluded_end_clopen)
{
    dyadic_interval di;
    dyadic expected{1, 0};
    ASSERT_TRUE(dyadic_equals(di.dexcluded_end(), expected));
}

TEST(dyadic_intervals, test_dincluded_end_opencl)
{
    dyadic_interval di{interval_type::opencl};
    dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dincluded_end(), expected));
}

TEST(dyadic_intervals, test_dexcluded_end_opencl)
{
    dyadic_interval di{interval_type::opencl};
    dyadic expected{-1, 0};
    ASSERT_TRUE(dyadic_equals(di.dexcluded_end(), expected));
}

TEST(dyadic_intervals, test_included_end_clopen)
{
    dyadic_interval di;

    ASSERT_EQ(di.included_end(), 0.0);
}

TEST(dyadic_intervals, test_excluded_end_clopen)
{
    dyadic_interval di;

    ASSERT_EQ(di.excluded_end(), 1.0);
}

TEST(dyadic_intervals, test_included_end_opencl)
{
    dyadic_interval di{interval_type::opencl};

    ASSERT_EQ(di.included_end(), 0.0);
}

TEST(dyadic_intervals, test_excluded_end_opencl)
{
    dyadic_interval di{interval_type::opencl};

    ASSERT_EQ(di.excluded_end(), -1.0);
}

TEST(dyadic_intervals, test_dsup_clopen)
{
    dyadic_interval di;
    dyadic expected{1, 0};
    ASSERT_TRUE(dyadic_equals(di.dsup(), expected));
}

TEST(dyadic_intervals, test_dinf_clopen)
{
    dyadic_interval di;
    dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dinf(), expected));
}

TEST(dyadic_intervals, test_dsup_opencl)
{
    dyadic_interval di{interval_type::opencl};
    dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dsup(), expected));
}

TEST(dyadic_intervals, test_dinf_opencl)
{
    dyadic_interval di{interval_type::opencl};
    dyadic expected{-1, 0};
    ASSERT_TRUE(dyadic_equals(di.dinf(), expected));
}

TEST(dyadic_intervals, test_sup_clopen)
{
    dyadic_interval di;

    ASSERT_EQ(di.sup(), 1.0);
}

TEST(dyadic_intervals, test_inf_clopen)
{
    dyadic_interval di;

    ASSERT_EQ(di.inf(), 0.0);
}

TEST(dyadic_intervals, test_sup_opencl)
{
    dyadic_interval di{interval_type::opencl};

    ASSERT_EQ(di.sup(), 0.0);
}

TEST(dyadic_intervals, test_inf_opencl)
{
    dyadic_interval di{interval_type::opencl};

    ASSERT_EQ(di.inf(), -1.0);
}


TEST(dyadic_intervals, test_flip_interval_aligned_clopen)
{

    dyadic_interval di{dyadic{0, 1}};
    dyadic_interval expected{dyadic{1, 1}};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(dyadic_intervals, test_flip_interval_non_aligned_clopen)
{

    dyadic_interval di{dyadic{1, 1}};
    dyadic_interval expected{dyadic{0, 1}};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(dyadic_intervals, test_flip_interval_aligned_opencl)
{

    dyadic_interval di{dyadic{0, 1}, interval_type::opencl};
    dyadic_interval expected{dyadic{-1, 1}, interval_type::opencl};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(dyadic_intervals, test_flip_interval_non_aligned_opencl)
{

    dyadic_interval di{dyadic{1, 1}, interval_type::opencl};
    dyadic_interval expected{dyadic{2, 1}, interval_type::opencl};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(dyadic_intervals, test_aligned_aligned)
{
    dyadic_interval di{dyadic{0, 0}};

    ASSERT_TRUE(di.aligned());
}

TEST(dyadic_intervals, test_aligned_non_aligned)
{
    dyadic_interval di{dyadic{1, 0}};

    ASSERT_FALSE(di.aligned());
}

TEST(dyadic_intervals, test_contains_unit_and_half)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{0, 1}};

    ASSERT_TRUE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_and_half_compliment)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{1, 1}};

    ASSERT_TRUE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_and_unit)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{0, 0}};

    ASSERT_TRUE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_and_longer)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{0, -1}};

    ASSERT_FALSE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_disjoint)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{1, 0}};

    ASSERT_FALSE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_disjoint_and_shorter_right)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{2, 1}};

    ASSERT_FALSE(parent.contains(child));
}

TEST(dyadic_intervals, test_contains_unit_disjoint_and_shorter_left)
{
    dyadic_interval parent{dyadic{0, 0}}, child{dyadic{-1, 1}};

    ASSERT_FALSE(parent.contains(child));
}


TEST(dyadic_intervals, test_to_dyadic_intervals_unit_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(0.0, 1.0, 1);
    dyadic_interval expected{dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 1);
    ASSERT_EQ(intervals[0], expected);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_unit_interval_tol_5)
{

    auto intervals = to_dyadic_intervals(0.0, 1.0, 5);
    dyadic_interval expected{dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 1);
    ASSERT_EQ(intervals[0], expected);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_mone_one_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(-1.0, 1.0, 1);
    dyadic_interval expected0{dyadic{-1, 0}}, expected1{dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}


TEST(dyadic_intervals, test_to_dyadic_intervals_mone_onehalf_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(-1.0, 1.5, 1);
    dyadic_interval expected0{dyadic{-1, 0}},
            expected1{dyadic{0, 0}},
            expected2{dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_mone_onequarter_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(-1.0, 1.25, 1);
    dyadic_interval expected0{dyadic{-1, 0}},
            expected1{dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_mone_onequarter_interval_tol_2)
{

    auto intervals = to_dyadic_intervals(-1.0, 1.25, 2);
    dyadic_interval expected0{dyadic{-1, 0}},
            expected1{dyadic{0, 0}},
            expected2{dyadic{4, 2}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}


TEST(dyadic_intervals, test_to_dyadic_intervals_0_upper_interval_tol_1)
{
    auto intervals = to_dyadic_intervals(0.0, 1.63451, 1);
    dyadic_interval expected0{dyadic{0, 0}},
            expected1{dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_0_upper_interval_tol_2)
{
    auto intervals = to_dyadic_intervals(0.0, 1.63451, 2);
    dyadic_interval expected0{dyadic{0, 0}},
            expected1{dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_0_upper_interval_tol_3)
{
    auto intervals = to_dyadic_intervals(0.0, 1.63451, 3);
    dyadic_interval expected0{dyadic{0, 0}},
            expected1{dyadic{2, 1}},
            expected2{dyadic{12, 3}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}

TEST(dyadic_intervals, test_to_dyadic_intervals_0_upper_interval_tol_7)
{
    auto intervals = to_dyadic_intervals(0.0, 1.63451, 7);
    dyadic_interval expected0{dyadic{0, 0}},
            expected1{dyadic{2, 1}},
            expected2{dyadic{12, 3}},
            expected3{dyadic{208, 7}};

    ASSERT_EQ(intervals.size(), 4);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
    ASSERT_EQ(intervals[3], expected3);
}

TEST(dyadic_intervals, shrink_interval_left_clopen)
{
    dyadic_interval start{dyadic{0, 0}};

    dyadic_interval expected{dyadic{0, 1}};

    ASSERT_EQ(start.shrink_interval_left(), expected);
}

TEST(dyadic_intervals, shrink_interval_right_clopen)
{
    dyadic_interval start{dyadic{0, 0}};

    dyadic_interval expected{dyadic{1, 1}};

    ASSERT_EQ(start.shrink_interval_right(), expected);
}

TEST(dyadic_intervals, shrink_interval_left_opencl)
{
    dyadic_interval start{dyadic{0, 0}, interval_type::opencl};

    dyadic_interval expected{dyadic{-1, 1}, interval_type::opencl};

    ASSERT_EQ(start.shrink_interval_left(), expected);
}

TEST(dyadic_intervals, shrink_interval_right_opencl)
{
    dyadic_interval start{dyadic{0, 0}, interval_type::opencl};

    dyadic_interval expected{dyadic{0, 1}, interval_type::opencl};

    ASSERT_EQ(start.shrink_interval_right(), expected);
}

TEST(dyadic_intervals, shrink_to_contained_end_clopen)
{
    dyadic_interval start{dyadic{0, 0}};

    dyadic_interval expected{dyadic{0, 1}};

    ASSERT_EQ(start.shrink_to_contained_end(), expected);
}

TEST(dyadic_intervals, shrink_to_omitted_end_clopen)
{
    dyadic_interval start{dyadic{0, 0}};

    dyadic_interval expected{dyadic{1, 1}};

    ASSERT_EQ(start.shrink_to_omitted_end(), expected);
}

TEST(dyadic_intervals, shrink_to_contained_end_opencl)
{
    dyadic_interval start{dyadic{0, 0}, interval_type::opencl};

    dyadic_interval expected{dyadic{0, 1}, interval_type::opencl};

    ASSERT_EQ(start.shrink_to_contained_end(), expected);
}

TEST(dyadic_intervals, shrink_to_omitted_end_opencl)
{
    dyadic_interval start{dyadic{0, 0}, interval_type::opencl};

    dyadic_interval expected{dyadic{-1, 1}, interval_type::opencl};

    ASSERT_EQ(start.shrink_to_omitted_end(), expected);
}
