cmake_minimum_required(VERSION 3.16)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(esig_common_tests)
target_link_libraries(esig_common_tests PRIVATE
        GTest::gtest_main
        esig_common
        )
target_sources(esig_common_tests
        PRIVATE
            test_dyadic.cpp
            test_dyadic_interval.cpp
        )


gtest_discover_tests(esig_common_tests)
