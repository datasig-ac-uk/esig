
cmake_minimum_required(VERSION 3.16)


project(esig_algebra VERSION 0.1)


find_package(pybind11 CONFIG REQUIRED)
include(GNUInstallDirs)
include(GenerateExportHeader)


pybind11_add_module(esig_algebra SHARED )
# Use add_library because it works better with the IDE
#add_library(esig_algebra SHARED)
#pybind11_extension(esig_algebra)
#target_link_libraries(esig_algebra PRIVATE
#        pybind11::headers
#        PUBLIC
#        pybind11::module
#        Python::Module
#        )

target_link_libraries(esig_algebra PUBLIC esig_common)
set_target_properties(esig_algebra PROPERTIES
        LIBRARY_OUTPUT_NAME algebra
        INSTALL_RPATH $ORIGIN)

generate_export_header(esig_algebra
        EXPORT_FILE_NAME "include/esig/algebra/esig_algebra_export.h")


target_include_directories(esig_algebra PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )


target_sources(esig_algebra
        PRIVATE
            src/py_algebra/py_esig_algebra.cpp
            src/basis.cpp
            src/free_tensor_interface_fused_default.cpp
            src/lie_interface_fused_default.cpp
            src/lie_basis/hall_set.h
            src/lie_basis/hall_set.cpp
            src/lie_basis/hall_extension.h
            src/lie_basis/hall_basis_map.h
            src/lie_basis/hall_basis_map.cpp
            src/lie_basis/lie_basis.h
            src/lie_basis/lie_basis.cpp
            src/tensor_basis/tensor_basis.h
            src/tensor_basis/tensor_basis.cpp
            src/fallback_context.h
            src/fallback_context.cpp
            src/free_tensor_wrapper.cpp
            src/context.cpp
            src/dense_lie.h
            src/dense_lie.cpp
            src/dense_tensor.h
            src/dense_tensor.cpp
            src/sparse_tensor.h
            src/sparse_tensor.cpp
            src/sparse_lie.h
            src/sparse_lie.cpp
            src/lie_wrapper.cpp
            src/iteration.cpp
            src/dense_kv_iterator.h
            src/sparse_kv_iterator.h
            src/sparse_mutable_reference.h
            src/coefficient_proxies.cpp
            src/py_algebra/py_lie.cpp
            src/py_algebra/py_lie.h
            src/py_algebra/py_coefficients.h
            src/py_algebra/py_coefficients.cpp
            src/py_algebra/py_tensor_key.cpp
            src/py_algebra/py_free_tensor_iterator.cpp
            src/py_algebra/py_free_tensor.cpp
            src/py_algebra/py_free_tensor.h
            src/py_algebra/py_lie_key.cpp
            src/py_algebra/py_lie_iterator.cpp
            src/py_algebra/py_tensor_key_iterator.h
            src/py_algebra/py_tensor_key_iterator.cpp
            src/py_algebra/py_lie_key_iterator.h
            src/py_algebra/py_lie_key_iterator.cpp
            src/coefficients.cpp
            src/data_buffer.cpp
            src/allocating_data_buffer.cpp
            src/rowed_data_buffer.cpp
            src/vector_construction_data.cpp
            src/stream_out_helpers.cpp
        PUBLIC
            include/esig/algebra/algebra_traits.h
            include/esig/algebra/basis.h
            include/esig/algebra/free_tensor_interface.h
            include/esig/algebra/lie_interface.h
            include/esig/algebra/context.h
            include/esig/algebra/iteration.h
            include/esig/algebra/coefficients.h
        )


install(TARGETS esig_algebra
        LIBRARY DESTINATION esig
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/esig/algebra/esig_algebra_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/esig/algebra)



enable_testing()
add_subdirectory(tests)
