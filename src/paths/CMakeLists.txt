cmake_minimum_required(VERSION 3.16)

project(esig_paths VERSION 0.1)


find_package(pybind11 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS container)

include(GNUInstallDirs)
include(GenerateExportHeader)



pybind11_add_module(esig_paths SHARED)

#add_library(esig_paths SHARED )
#pybind11_extension(esig_paths)


generate_export_header(esig_paths
        EXPORT_FILE_NAME "include/esig/paths/esig_paths_export.h"
        )


target_link_libraries(esig_paths
        PUBLIC
            esig_algebra
        PRIVATE
            Boost::boost
            Boost::container
        )

set_target_properties(esig_paths PROPERTIES
        LIBRARY_OUTPUT_NAME _paths
        INSTALL_RPATH $ORIGIN
        )

target_include_directories(esig_paths PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )



target_sources(esig_paths
        PRIVATE
            src/py_paths/py_esig_paths.cpp
            src/path.cpp
            src/lie_increment_path.h
            src/lie_increment_path.cpp
            src/dynamically_generated_path.h
            src/dynamically_generated_path.cpp
            src/py_paths/py_path.h
            src/py_paths/py_path.cpp
            src/py_paths/py_lie_increment_path.cpp
            src/py_paths/py_lie_increment_path.h
            src/py_paths/py_function_path.cpp
            src/py_paths/py_function_path.h
            src/py_paths/py_tick_path.h
            src/py_paths/py_tick_path.cpp
            src/dyadic_caching_layer.cpp
            src/tick_path.h
            src/tick_path.cpp
        PUBLIC
            include/esig/paths/path.h
            include/esig/paths/python_interface.h
        )

install(TARGETS esig_paths
        LIBRARY DESTINATION esig
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/esig_paths_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/esig)
install(FILES
        __init__.py
        cloud.py
        DESTINATION esig/paths_module
        )
install(FILES paths.py DESTINATION esig)

enable_testing()
add_subdirectory(tests)
