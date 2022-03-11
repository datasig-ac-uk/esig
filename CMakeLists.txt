cmake_minimum_required(VERSION 3.16)
project(esig)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
#set(CMAKE_CXX_DEFAULT_VISIBILITY hidden)

if (WIN32)
    set(Boost_NO_BOOST_CMAKE ON)
endif()


find_package(Boost COMPONENTS system thread REQUIRED)

message(STATUS "Boost thread library: ${Boost_THREAD_LIBRARY_RELEASE}")
message(STATUS "Boost system library: ${Boost_SYSTEM_LIBRARY_RELEASE}")

if(NOT EXISTS "${Boost_THREAD_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost thread library not found")
endif()

if (NOT EXISTS "${Boost_SYSTEM_LIBRARY_RELEASE}")
    message(FATAL_ERROR "Boost system library not found")
endif ()


set(Python_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv")
if (SKBUILD)

    find_package(PythonExtensions REQUIRED)
    find_package(NumPy REQUIRED)


    add_library(tosig MODULE)
    python_extension_module(tosig)

    message(STATUS "Generating switch.h")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/switch_generator.py" "${CMAKE_CURRENT_SOURCE_DIR}/src")

    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    target_include_directories(tosig PRIVATE ${NumPy_INCLUDE_DIR})

else()


    # Look for Virtual environments first
    set(Python_FIND_VIRTUALENV FIRST)

    message(STATUS "FIND_VIRTUALENV: ${Python_FIND_VIRTUALENV}")
    find_package(Python 3 REQUIRED
            COMPONENTS
            Interpreter
            Development
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
endif()


if(EXISTS "${CMAKE_SOURCE_DIR}/build/recombine")
#    add_subdirectory("${CMAKE_SOURCE_DIR}/build/recombine")
    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
else()
    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
endif()

if(MSVC)
    target_compile_options(tosig PRIVATE /bigobj)
endif()

target_link_libraries(tosig Boost::system Boost::boost Boost::thread)

target_sources(tosig PRIVATE
        src/Cpp_ToSig.cpp
        src/stdafx.h
        src/switch.h
        src/ToSig.cpp
        src/ToSig.h
        src/tosig_module.cpp)

#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")




target_include_directories(tosig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libalgebra")


install(TARGETS tosig LIBRARY DESTINATION  "esig")



