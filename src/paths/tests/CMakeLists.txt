cmake_minimum_required(VERSION 3.16)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(esig_paths_tests)
target_link_libraries(esig_paths_tests PRIVATE GTest::gtest_main esig_paths)

target_sources(esig_paths_tests PRIVATE
            test_path_wrapper.cpp
        )

find_package(Boost CONFIG REQUIRED COMPONENTS container)

add_executable(esig_paths_lie_increment_path_tests)
target_link_libraries(esig_paths_lie_increment_path_tests
        PRIVATE GTest::gtest_main esig_paths Boost::container)

target_sources(esig_paths_lie_increment_path_tests
        PRIVATE
            lie_increment_path_tests.cpp
            ../src/lie_increment_path.h
            ../src/lie_increment_path.cpp
        )

add_executable(esig_paths_tick_path_tests
        test_tick_path.cpp
        ../src/tick_path.cpp
        ../src/tick_path.h
        )
target_link_libraries(esig_paths_tick_path_tests PRIVATE
        GTest::gtest_main esig_paths Boost::container)

gtest_discover_tests(esig_paths_tests)
gtest_discover_tests(esig_paths_lie_increment_path_tests)
gtest_discover_tests(esig_paths_tick_path_tests)
