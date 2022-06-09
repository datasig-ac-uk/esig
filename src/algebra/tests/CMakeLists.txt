cmake_minimum_required(VERSION 3.16)


find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(esig_algebra_tests )
target_link_libraries(esig_algebra_tests PRIVATE
        GTest::gtest_main
        GTest::gmock
        esig_algebra
        )

target_include_directories(esig_algebra_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)

target_sources(esig_algebra_tests PRIVATE
            fixture_base.h
            fixture_base.cpp
            test_tensor.cpp
            test_lie.cpp
            test_lie_interface.cpp
            test_tensor_interface.cpp
            mock_context.h
            test_context.cpp
        )


add_executable(esig_algebra_test_tensor
        test_dense_tensor.cpp
        test_sparse_tensor.cpp
        la_esig_equality_helper.h
        ../src/dense_tensor.cpp
        ../src/sparse_tensor.cpp
        ../src/tensor_basis/tensor_basis.cpp
        ../src/tensor_basis/tensor_basis.h
        ../src/basis.cpp
        )
target_link_libraries(esig_algebra_test_tensor PRIVATE
        GTest::gtest_main
        esig_common
        esig_algebra
        Libalgebra::Libalgebra
        )
target_include_directories(esig_algebra_test_tensor PRIVATE
        ../src
        ../include
        ${CMAKE_CURRENT_BINARY_DIR}/../include
        )

add_executable(esig_algebra_test_lie
        test_dense_lie.cpp
        test_sparse_lie.cpp
        la_esig_equality_helper.h
        ../src/lie_basis/hall_basis_map.cpp
        ../src/lie_basis/hall_set.cpp
        ../src/lie_basis/lie_basis.cpp
        ../src/basis.cpp
        )
target_link_libraries(esig_algebra_test_lie PRIVATE
        GTest::gtest_main
        esig_common
        esig_algebra
        Libalgebra::Libalgebra
        )
target_include_directories(esig_algebra_test_lie PRIVATE
        ../src
        ../include
        ${CMAKE_CURRENT_BINARY_DIR}/../include
        )

add_executable(esig_algebra_test_lie_basis
        test_lie_basis.cpp
        ../src/lie_basis/lie_basis.cpp
        ../src/lie_basis/hall_basis_map.cpp
        ../src/lie_basis/hall_set.cpp
        ../src/basis.cpp
        )
target_link_libraries(esig_algebra_test_lie_basis PRIVATE
        GTest::gtest_main
        esig_common
        esig_algebra
        Libalgebra::Libalgebra)
target_include_directories(esig_algebra_test_lie_basis PRIVATE
        ../src/lie_basis
        ../include
        ${CMAKE_CURRENT_BINARY_DIR}/../include)



gtest_discover_tests(esig_algebra_tests)
gtest_discover_tests(esig_algebra_test_tensor)
gtest_discover_tests(esig_algebra_test_lie)
gtest_discover_tests(esig_algebra_test_lie_basis)
