cmake_minimum_required(VERSION 3.21)
project(esig)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
set(CMAKE_CXX_DEFAULT_VISIBILITY hidden)

# Installing needs special handling here because the target is ultimately
# a Python wheel rather than a standard directory structure.
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/esig" )
set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/esig" )
set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/esig/include" )
if (WIN32)
    set(RECOMBINE_INSTALL_DEPENDENTS ON)
endif()

include(GNUInstallDirs)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)


find_package(Boost COMPONENTS system thread REQUIRED)

message(STATUS "Boost thread library: ${Boost_THREAD_LIBRARY_RELEASE}")
message(STATUS "Boost system library: ${Boost_SYSTEM_LIBRARY_RELEASE}")

if(NOT EXISTS "${Boost_THREAD_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost thread library not found")
endif()

if (NOT EXISTS "${Boost_SYSTEM_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost system library not found")
endif ()


if (SKBUILD)
    message(STATUS "skbuild environment: ${PYTHON_EXECUTABLE}")
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    set(Python_LIBRARY "${PYTHON_LIBRARY}")
    cmake_path(REMOVE_FILENAME PYTHON_EXECUTABLE OUTPUT_VARIABLE PYTHON_EXE)
    cmake_path(APPEND PYTHON_EXE "../")
    cmake_path(NORMAL_PATH PYTHON_EXE)
    set(Python_ROOT_DIR "${PYTHON_EXE}")
    unset(PYTHON_EXE)

    set(Python_FIND_STRATEGY LOCATION)
    set(Python_FIND_UNVERSIONED_NAMES FIRST)


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

message(STATUS "Generating switch.h")
execute_process(COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/switch_generator.py" "${CMAKE_CURRENT_SOURCE_DIR}/src")


Python_add_library(tosig MODULE WITH_SOABI)
target_link_libraries(tosig PRIVATE Python::NumPy)


target_sources(tosig PRIVATE
        src/Cpp_ToSig.cpp
        src/stdafx.h
        src/switch.h
        src/ToSig.cpp
        src/ToSig.h
        src/tosig_module.cpp)


#find_package(recombine CONFIG QUIET))
set(RECOMBINE_MAKE_FRAMEWORK OFF)

execute_process(COMMAND "${Python_EXECUTABLE}" -c "import platform; print(platform.architecture()[0])"
        OUTPUT_VARIABLE arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
if (arch STREQUAL "64bit")
    set(RECOMBINE_ARCH x86_64)
else()
    set(RECOMBINE_ARCH x86)
endif()





if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/build/recombine/")
    message(STATUS "Recombine found")
    add_subdirectory(build/recombine)
    target_link_libraries(tosig PRIVATE recombine)
    target_include_directories(tosig PRIVATE ${CMAKE_SOURCE_DIR}/recombine)
    target_sources(tosig PRIVATE recombine/_recombine.cpp recombine/TestVec/RdToPowers2.cpp)

    #    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
else()
    message(STATUS "Recombine not found")
    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
endif()


if(MSVC)
    target_compile_options(tosig PRIVATE /bigobj)
endif()


target_link_libraries(tosig PRIVATE Boost::system Boost::boost Boost::thread Boost::dynamic_linking)


#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")

set_target_properties(tosig PROPERTIES
        LIBRARY_OUTPUT_NAME _tosig
        )


target_include_directories(tosig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libalgebra")


install(TARGETS tosig
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/esig
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
#install(FILES ${CMAKE_CURRENT_LIST_DIR}/esig/*.py DESTINATION ${CMAKE_INSTALL_PREFIX}/esig)