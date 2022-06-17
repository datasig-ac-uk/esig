cmake_minimum_required(VERSION 3.21)

find_package(GTest CONFIG)
if (TARGET GTest::gtest)
include(GoogleTest)

add_executable(esig_la_context_tests)
target_link_libraries(esig_la_context_tests PRIVATE
        GTest::gtest_main
        esig_common
        esig_algebra
        esig_libalgebra_context
        Libalgebra::Libalgebra
        )


target_sources(esig_la_context_tests
        PRIVATE
            test_libalgebra_contexts_available.cpp
        )


gtest_discover_tests(esig_la_context_tests)
endif()
