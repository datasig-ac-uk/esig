
cmake_minimum_required(VERSION 3.16)


project(esig_algebra VERSION 0.1)


include(GNUInstallDirs)
include(GenerateExportHeader)

add_library(esig_algebra SHARED)
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
            Libalgebra_lite
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

set_target_properties(esig_algebra PROPERTIES
        PUBLIC_HEADER "${esig_algebra_HEADERS}"
        LIBRARY_OUTPUT_DIRECTORY "${ESIG_BUILD_PYMODULE}"
        POSITION_INDEPENDENT_CODE ON
        )

target_sources(esig_algebra
        PRIVATE
            src/basis.cpp
            src/free_tensor_wrapper.cpp
            src/context.cpp
            src/lie_wrapper.cpp
            src/iteration.cpp
            src/dense_data_access_iterator.cpp
            src/detail/converting_lie_iterator_adaptor.h
            src/shuffle_tensor.cpp
            src/algebra_base.cpp
            src/basis_interface.cpp
            src/basis_interface.h
            src/basis_object.cpp
            src/lite_context.cpp
            src/lite_context.h
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
