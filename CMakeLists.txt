cmake_minimum_required(VERSION 3.16)
project(esig)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
#set(CMAKE_CXX_DEFAULT_VISIBILITY hidden)



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

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)"
            OUT_VARIABLE Python_ROOT_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )

else()
    # Look for Virtual environments first
    set(Python_FIND_VIRTUALENV FIRST)
    if (EXISTS ${CMAKE_SOURCE_DIR}/venv)
        set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/venv")
    endif()

endif()

find_package(Python REQUIRED
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

include(cmake/findPyMKL.cmake)

Python_add_library(tosig MODULE WITH_SOABI)
target_link_libraries(tosig PRIVATE Python::NumPy)


target_sources(tosig PRIVATE
        src/Cpp_ToSig.cpp
        src/stdafx.h
        src/switch.h
        src/ToSig.cpp
        src/ToSig.h
        src/tosig_module.cpp)

if(EXISTS "${CMAKE_SOURCE_DIR}/build/recombine")
    #    add_subdirectory("${CMAKE_SOURCE_DIR}/build/recombine")
    add_subdirectory(build/recombine)
    target_link_libraries(tosig PRIVATE recombine)
    target_include_directories(tosig PRIVATE
            ${CMAKE_SOURCE_DIR}/recombine
            )

    target_sources(tosig PRIVATE recombine/_recombine.cpp recombine/TestVec/RdToPowers2.cpp)

    #    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
else()
    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
endif()
if(MSVC)
    target_compile_options(tosig PRIVATE /bigobj)
endif()


target_link_libraries(tosig PRIVATE Boost::system Boost::boost Boost::thread Boost::dynamic_linking)


#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")

set_target_properties(tosig PROPERTIES
        LIBRARY_OUTPUT_NAME _tosig
        INSTALL_RPATH "$ORIGIN/lib"
        )


target_include_directories(tosig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libalgebra")


install(TARGETS tosig DESTINATION ${CMAKE_SOURCE_DIR}/esig)



