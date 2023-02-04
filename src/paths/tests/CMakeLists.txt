cmake_minimum_required(VERSION 3.16)

find_package(GTest CONFIG)
if (TARGET GTest::gtest)

include(GoogleTest)

add_executable(esig_paths_tests)
target_link_libraries(esig_paths_tests PRIVATE
        GTest::gtest_main
        esig_common
        esig_paths)

target_sources(esig_paths_tests PRIVATE
            test_path_wrapper.cpp
            lie_increment_path_tests.cpp
            RandomScalarsFixture.cpp
            RandomScalarsFixture.h
        )




gtest_discover_tests(esig_paths_tests)
endif()
