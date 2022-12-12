cmake_minimum_required(VERSION 3.21)
project(esig VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
set(CMAKE_CXX_DEFAULT_VISIBILITY hidden)
include(GNUInstallDirs)
# Installing needs special handling here because the target is ultimately
# a Python wheel rather than a standard directory structure.
set(CMAKE_INSTALL_LIBDIR "src/esig" )
set(CMAKE_INSTALL_BINDIR "src/esig" )
set(CMAKE_INSTALL_INCLUDEDIR "src/esig/include" )
set(RECOMBINE_INSTALL_DEPENDENTS ON)

option(ESIG_BUILD_TESTS "Build C++ tests" FALSE)

if(APPLE)
#    set(CMAKE_SKIP_BUILD_RPATH OFF)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()


#list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)



find_package(Boost 1.75 COMPONENTS headers system thread REQUIRED)

message(STATUS "Boost thread library: ${Boost_THREAD_LIBRARY_RELEASE}")
message(STATUS "Boost system library: ${Boost_SYSTEM_LIBRARY_RELEASE}")

if(NOT EXISTS "${Boost_THREAD_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost thread library not found")
endif()

if (NOT EXISTS "${Boost_SYSTEM_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost system library not found")
endif ()

set(PYBIND11_FINDPYTHON ON)
if (NOT PYTHON_FOUND AND SKBUILD)
#    set(Python_FIND_ABI "ANY" "ANY" "ANY")

    message(STATUS "skbuild environment: ${PYTHON_EXECUTABLE}")
#    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE INTERNAL "")
#    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}" CACHE INTERNAL "")
#    set(Python_LIBRARY "${PYTHON_LIBRARY}" CACHE INTERNAL "")
#    cmake_path(REMOVE_FILENAME PYTHON_EXECUTABLE OUTPUT_VARIABLE PYTHON_EXE)
#    cmake_path(APPEND PYTHON_EXE "../")
#    cmake_path(NORMAL_PATH PYTHON_EXE)
#    set(Python_ROOT_DIR "${PYTHON_EXE}")
#    unset(PYTHON_EXE)

#    set(Python_FIND_STRATEGY LOCATION)
#    set(Python_FIND_UNVERSIONED_NAMES FIRST)

    unset(Python_LIBRARY CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(Python3_LIBRARY CACHE)

else()
    # Look for Virtual environments first
    set(Python_FIND_VIRTUALENV FIRST)
    if (EXISTS ${CMAKE_SOURCE_DIR}/venv)
        set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/venv")
    endif()

endif()

message(STATUS "Python root: ${Python_ROOT_DIR}")
find_package(Python 3 REQUIRED
        COMPONENTS
        Interpreter
        Development.Module
        NumPy
        )

message(STATUS "Found Python: ${Python_VERSION}")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python directory: ${Python_INCLUDE_DIRS}")
message(STATUS "Numpy version: ${Python_NumPy_VERSION}")
message(STATUS "Numpy includes: ${Python_NumPy_INCLUDE_DIRS}")

#find_package(pybind11 CONFIG REQUIRED)

#find_package(recombine CONFIG QUIET))


execute_process(COMMAND "${Python_EXECUTABLE}" -c "import platform; print(platform.architecture()[0])"
        OUTPUT_VARIABLE arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
if (arch STREQUAL "64bit")
    set(RECOMBINE_ARCH x86_64)
else ()
    set(RECOMBINE_ARCH x86)
endif ()

set(RECOMBINE_MAKE_FRAMEWORK OFF CACHE BOOL "Do not make a framework on Apple")



#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")



# Add dependencies
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/libalgebra)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/recombine)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/pybind11)

enable_testing()
# Add the actual source folders
add_subdirectory(src)



add_custom_target(esig DEPENDS
        esig_common
        esig_algebra
        esig_paths
        esig_py_module
        )
