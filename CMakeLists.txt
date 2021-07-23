cmake_minimum_required(VERSION 3.19)
project(esig)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
#set(CMAKE_CXX_DEFAULT_VISIBILITY hidden)



set(Python_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv")

# Look for Virtual environments first
set(Python_FIND_VIRTUALENV FIRST)

message(STATUS "FIND_VIRTUALENV: ${Python_FIND_VIRTUALENV}")
find_package(Python 3.5 REQUIRED
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

#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libalgebra")

message(STATUS "Generating switch.h")
execute_process(COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/switch_generator.py" "${CMAKE_CURRENT_SOURCE_DIR}/src")



Python_add_library(tosig MODULE WITH_SOABI
        src/Cpp_ToSig.cpp
        src/stdafx.h
        src/switch.h
        src/ToSig.cpp
        src/ToSig.h
        src/tosig_module.cpp)


target_compile_definitions(tosig PRIVATE ESIG_NO_RECOMBINE)

target_include_directories(tosig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libalgebra")
target_link_libraries(tosig PRIVATE Python::NumPy)


install(TARGETS tosig DESTINATION  "${CMAKE_CURRENT_SOURCE_DIR}/esig")



