cmake_minimum_required(VERSION 3.16)
message(STATUS "Esig common library VERSION ${esig_VERSION}")
message(STATUS "MAJOR ${esig_VERSION_MAJOR}")
message(STATUS "MINOR ${esig_VERSION_MINOR}")
message(STATUS "PATCH ${esig_VERSION_PATCH}")
message(STATUS "TWEAK ${esig_VERSION_TWEAK}")

project(esig_common VERSION 1.0.0)

include(GNUInstallDirs)
include(GenerateExportHeader)

find_package(Boost CONFIG REQUIRED)
find_package(Bignum QUIET)

add_library(esig_common SHARED src/scalars/conversion_functions.cpp)
#add_library(esig_common SHARED )
#pybind11_extension(esig_common)
#target_link_libraries(esig_common PUBLIC
#        pybind11::module
#        )
generate_export_header(esig_common
        BASE_NAME "esig"
        EXPORT_FILE_NAME "esig_export.h")

set(esig_common_HEADERS
        "include/esig/implementation_types.h"
        "include/esig/intervals.h"
        "include/esig/scalars.h"
        "${CMAKE_CURRENT_BINARY_DIR}/esig_export.h")


set_target_properties(esig_common PROPERTIES
        PUBLIC_HEADER "${esig_common_HEADERS}"
        LIBRARY_OUTPUT_DIRECTORY ${ESIG_BUILD_PYMODULE}
        )

target_sources(esig_common
        PRIVATE
            src/intervals.cpp
            src/segmentation.h
            src/segmentation.cpp
            src/dyadic.cpp
            src/dyadic_interval.cpp
            src/to_dyadic_intervals.cpp
            src/real_interval.cpp
            src/interval_partition.cpp
            src/scalars/scalar_type.cpp
            src/scalars/scalar_interface.cpp
            src/scalars/scalar.cpp
            src/scalars/scalar_pointer.cpp
            src/scalars/scalar_array.cpp
            src/scalars/float_type.cpp
            src/scalars/float_type.h
            src/scalars/double_type.cpp
            src/scalars/double_type.h
            src/scalars/float_scalar_type_holder.cpp
            src/scalars/double_scalar_type_holder.cpp
            src/scalars/owned_scalar_array.cpp
            src/scalars/type_registry.cpp
            src/scalars/scalar_stream.cpp
            src/scalars/standard_scalar_type.h
            src/scalars/rational_type.cpp
            src/scalars/rational_type.h
            src/scalars/ratonal_scalar_type_holder.cpp
            src/scalars/key_scalar_array.cpp
            ${esig_common_HEADERS}
        )


target_link_libraries(esig_common PRIVATE Boost::boost)
if (TARGET Bignum::Bignum)
    set(ESIG_RATIONAL_HEADER "gmp.hpp")
    set(ESIG_RATIONAL_TYPE "boost::multiprecision::mpq_rational")
    target_link_libraries(esig_common PRIVATE Bignum::Bignum)
else()
    set(ESIG_RATIONAL_HEADER "cpp_int.hpp")
    set(ESIG_RATIONAL_TYPE "boost::multiprecision::cpp_rational")
endif()

configure_file("src/config.h.in" "config.h" NO_SOURCE_PERMISSIONS @ONLY)

target_include_directories(esig_common PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

set(ESIG_COMMON_INCLUDE_POLYNOMIALS OFF)
if (ESIG_COMMON_INCLUDE_POLYNOMIALS AND TARGET Libalgebra::Libagebra)
    target_link_libraries(esig_common PRIVATE Libalgebra::Libalgebra)
    target_sources(esig_common PRIVATE
        src/scalars/poly_scalars.cpp
        src/scalars/poly_scalars.h
        src/scalars/poly_ring_type_holder.cpp)

endif()



install(TARGETS esig_common
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/config.h
        ${CMAKE_CURRENT_BINARY_DIR}/esig_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )



enable_testing()
add_subdirectory(tests)
