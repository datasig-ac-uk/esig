cmake_minimum_required(VERSION 3.16)

project(esig_paths VERSION 0.1)


find_package(Boost REQUIRED)

include(GNUInstallDirs)
include(GenerateExportHeader)



add_library(esig_paths SHARED)

#add_library(esig_paths SHARED )
#pybind11_extension(esig_paths)

set(esig_paths_HEADERS
        "include/esig/paths/path.h"
        )

generate_export_header(esig_paths
        EXPORT_FILE_NAME "include/esig/paths/esig_paths_export.h"
        )


target_link_libraries(esig_paths
        PUBLIC
            esig_algebra
        PRIVATE
            Boost::boost
        )

#set_target_properties(esig_paths PROPERTIES
#        INSTALL_RPATH $ORIGIN
#        )

target_include_directories(esig_paths PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )



target_sources(esig_paths
        PRIVATE
            src/path.cpp
            src/lie_increment_path.h
            src/lie_increment_path.cpp
            src/dynamically_generated_path.h
            src/dynamically_generated_path.cpp
            src/dyadic_caching_layer.cpp
            src/tick_path.h
            src/tick_path.cpp
            ${esig_paths_HEADERS}
        )


pybind11_add_module(py_esig_paths MODULE WITH_SOABI)
set_target_properties(py_esig_paths PROPERTIES
        LIBRARY_OUTPUT_NAME _paths
        )

target_sources(py_esig_paths PRIVATE
        src/py_paths/py_esig_paths.cpp
        src/py_paths/py_path.h
        src/py_paths/py_path.cpp
        src/py_paths/py_lie_increment_path.cpp
        src/py_paths/py_lie_increment_path.h
        src/py_paths/py_function_path.cpp
        src/py_paths/py_function_path.h
        src/py_paths/py_tick_path.h
        src/py_paths/py_tick_path.cpp
        )

target_include_directories(py_esig_paths PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(py_esig_paths PRIVATE esig_paths)


install(TARGETS esig_paths py_esig_paths
        LIBRARY DESTINATION esig
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/esig/paths/esig_paths_export.h
             "${CMAKE_CURRENT_LIST_DIR}/include/esig/paths/python_interface.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/esig)


enable_testing()
add_subdirectory(tests)
