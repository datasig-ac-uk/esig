
cmake_minimum_required(VERSION 3.16)


project(esig_algebra VERSION 0.1)


include(GNUInstallDirs)
include(GenerateExportHeader)

add_library(esig_algebra SHARED src/basis_interface.cpp src/basis_interface.h src/basis_object.cpp)
# Use add_library because it works better with the IDE
#add_library(esig_algebra SHARED)
#pybind11_extension(esig_algebra)
#target_link_libraries(esig_algebra PRIVATE
#        pybind11::headers
#        PUBLIC
#        pybind11::module
#        Python::Module
#        )

target_link_libraries(esig_algebra
        PUBLIC
            esig_common
        PRIVATE
            Boost::boost
        )
#set_target_properties(esig_algebra PROPERTIES
#        INSTALL_RPATH $ORIGIN)

generate_export_header(esig_algebra
        EXPORT_FILE_NAME "include/esig/algebra/esig_algebra_export.h")


target_include_directories(esig_algebra PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

set(esig_algebra_HEADERS
        "include/esig/algebra/algebra_traits.h"
        "include/esig/algebra/basis.h"
        "include/esig/algebra/tensor_interface.h"
        "include/esig/algebra/lie_interface.h"
        "include/esig/algebra/context.h"
        "include/esig/algebra/iteration.h"
        "include/esig/algebra/base.h"
        "include/esig/algebra/context_fwd.h"
        )

set_target_properties(esig_algebra PROPERTIES PUBLIC_HEADER "${esig_algebra_HEADERS}")

target_sources(esig_algebra
        PRIVATE
            src/basis.cpp
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
            src/fallback_maps.h
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
            src/dense_data_access_iterator.cpp
            src/fallback_maps.cpp
            src/detail/converting_lie_iterator_adaptor.h
            src/shuffle_tensor.cpp
            src/algebra_base.cpp
            ${esig_algebra_HEADERS}
        )



install(TARGETS esig_algebra
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/esig/algebra/esig_algebra_export.h"
        "${CMAKE_CURRENT_LIST_DIR}/include/esig/algebra/python_interface.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/esig/algebra)


if(ESIG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
