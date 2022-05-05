cmake_minimum_required(VERSION 3.21)
project(src/esig)

set(CMAKE_CXX_STANDARD 11)
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


if(APPLE)
#    set(CMAKE_SKIP_BUILD_RPATH OFF)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH @loader_path)
    set(CMAKE_INSTALL_NAME_DIR @rpath)
else()
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()


#list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)



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
#    set(Python_LIBRARY "${PYTHON_LIBRARY}")
#    cmake_path(REMOVE_FILENAME PYTHON_EXECUTABLE OUTPUT_VARIABLE PYTHON_EXE)
#    cmake_path(APPEND PYTHON_EXE "../")
#    cmake_path(NORMAL_PATH PYTHON_EXE)
#    set(Python_ROOT_DIR "${PYTHON_EXE}")
#    unset(PYTHON_EXE)

#    set(Python_FIND_STRATEGY LOCATION)
#    set(Python_FIND_UNVERSIONED_NAMES FIRST)


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
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/build/recombine/")
    message(STATUS "Recombine found")
    add_subdirectory(build/recombine)

    #    target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
    if(NOT APPLE)
        set_target_properties(recombine PROPERTIES
            NO_SONAME ON
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
else ()
    message(STATUS "Recombine not found")
    #target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)
endif ()

Python_add_library(tosig MODULE WITH_SOABI)
target_link_libraries(tosig PRIVATE Python::NumPy)


target_sources(tosig PRIVATE
        src/Cpp_ToSig.cpp
        src/stdafx.h
        src/switch.h
        src/ToSig.cpp
        src/ToSig.h
        src/tosig_module.cpp)


if(MSVC)
    target_compile_options(tosig PRIVATE /bigobj)
elseif(APPLE)
#    set_target_properties(tosig PROPERTIES INSTALL_RPATH @loader_path)
endif()

target_link_libraries(tosig PRIVATE Boost::system Boost::boost Boost::thread Boost::dynamic_linking)
if (TARGET recombine)
    target_link_libraries(tosig PRIVATE recombine)
    target_include_directories(tosig PRIVATE ${CMAKE_SOURCE_DIR}/recombine)
    target_sources(tosig PRIVATE recombine/_recombine.cpp recombine/TestVec/RdToPowers2.cpp)
    target_compile_definitions(tosig PRIVATE ESIG_WITH_RECOMBINE)
endif()
#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")

set_target_properties(tosig PROPERTIES
        LIBRARY_OUTPUT_NAME _tosig
        )


add_subdirectory(libalgebra)
target_link_libraries(tosig PRIVATE Libalgebra::Libalgebra)

install(TARGETS tosig
        LIBRARY DESTINATION src/esig
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
#install(FILES ${CMAKE_CURRENT_LIST_DIR}/esig/*.py DESTINATION ${CMAKE_INSTALL_PREFIX}/esig)

#install(FILES
#        "${CMAKE_CURRENT_LIST_DIR}/esig/__init__.py"
#        "${CMAKE_CURRENT_LIST_DIR}/esig/backends.py"
#        "${CMAKE_CURRENT_LIST_DIR}/esig/tosig.py"
#        "${CMAKE_CURRENT_LIST_DIR}/esig/VERSION"
#        DESTINATION esig
#        )
#install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/esig/tests" DESTINATION esig)