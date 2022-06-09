cmake_minimum_required(VERSION 3.16)
message(STATUS "Esig common library VERSION ${esig_VERSION}")
message(STATUS "MAJOR ${esig_VERSION_MAJOR}")
message(STATUS "MINOR ${esig_VERSION_MINOR}")
message(STATUS "PATCH ${esig_VERSION_PATCH}")
message(STATUS "TWEAK ${esig_VERSION_TWEAK}")

project(esig_common VERSION 1.0.0)

include(GNUInstallDirs)
include(GenerateExportHeader)
find_package(Python 3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)



pybind11_add_module(esig_common SHARED)
#add_library(esig_common SHARED )
#pybind11_extension(esig_common)
#target_link_libraries(esig_common PUBLIC
#        pybind11::module
#        )

generate_export_header(esig_common
        BASE_NAME esig
        EXPORT_FILE_NAME "include/esig/esig_export.h")

set_target_properties(esig_common PROPERTIES
        LIBRARY_OUTPUT_NAME common)

target_sources(esig_common
        PRIVATE
            src/py_common/py_common.cpp
            src/intervals.cpp
            src/segmentation.h
            src/segmentation.cpp
            src/dyadic.cpp
            src/dyadic_interval.cpp
            src/to_dyadic_intervals.cpp
            src/real_interval.cpp
            src/py_common/py_common.h
            src/py_common/py_real_interval.cpp
            src/py_common/py_intervals.cpp
            src/py_common/py_intervals.h
            src/py_common/py_dyadic.cpp
            src/py_common/py_dyadic_interval.cpp
            src/py_common/py_recombine.h
            src/py_common/py_recombine.cpp
            src/recombine/_recombine.h
            src/recombine/_recombine.cpp
            src/recombine/TestVec/RdToPowers2.cpp
            src/recombine/TestVec/RdToPowers.h
            src/recombine/TestVec/EvaluateAllMonomials.h
            src/recombine/TestVec/RdToPowers2.h
        PUBLIC
            include/esig/implementation_types.h
            include/esig/intervals.h
        )

configure_file("src/config.h.in" "include/esig/config.h" NO_SOURCE_PERMISSIONS @ONLY)

target_include_directories(esig_common PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )


install(TARGETS esig_common DESTINATION esig)
install(FILES
        __init__.py
        tosig.py
        DESTINATION esig
        )
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/esig/config.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/esig/esig_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )



enable_testing()
add_subdirectory(tests)
