cmake_minimum_required(VERSION 3.16)

#project(esig_libalgebra_context VERSION 0.1)

find_package(Boost REQUIRED)

if(DEFINED ESIG_LA_CONFIG)
    cmake_path(HAS_PARENT ESIG_LA_CONFIG has_parent)
    if (NOT has_parent)
        if (EXISTS ESIG_LA_CONFIG)
            cmake_path(ABSOLUTE ESIG_LA_CONFIG OUTPUT_VARIABLE la_config)
        else()
            cmake_path(REPLACE_EXTENSION ESIG_LA_CONFIG ".json")
            if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/config/${ESIG_LA_CONFIG}")
                set(la_config "${CMAKE_CURRENT_LIST_DIR}/config/${ESIG_LA_CONFIG}")
                cmake_path(ABSOLUTE la_config)
            else()
                message(FATAL_ERROR "The configuration ${ESIG_LA_CONFIG} cannot be found")
            endif()
        endif()
    else()
        cmake_path(ABSOLUTE ESIG_LA_CONFIG OUTPUT_VARIABLE la_config)
        if(NOT EXISTS la_config)
            message(FATAL_ERROR "The configuration ${la_config} does not exist")
        endif()
    endif()
else()
    set(la_config "${CMAKE_CURRENT_LIST_DIR}/config/default.json")
endif()
message(STATUS "Found LA config: ${la_config}")

execute_process(COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/tools/switch_builder.py"
        "-c${la_config}"
        "-o${CMAKE_CURRENT_BINARY_DIR}/la"
        OUTPUT_VARIABLE ESIG_LA_GENERATED_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

#add_library(esig_la_contexts OBJECT)
#
#set_target_properties(esig_la_contexts PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_include_directories(esig_la_contexts
#        PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/la"
#        PRIVATE "include"
#        )
#target_sources(esig_la_contexts PRIVATE ${ESIG_LA_GENERATED_FILES})
#target_link_libraries(esig_la_contexts PUBLIC
#        esig_algebra
#        Libalgebra::Libalgebra
#        Boost::boost
#        )



add_library(esig_libalgebra_context STATIC)

target_link_libraries(esig_libalgebra_context
        PRIVATE
        esig_algebra
        Libalgebra::Libalgebra
        Boost::boost
        )

set_target_properties(esig_libalgebra_context PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(esig_libalgebra_context
        PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/la"
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

target_sources(esig_libalgebra_context
        PUBLIC
        include/esig/libalgebra_context/libalgebra_context.h
        PRIVATE
        src/libalgebra_context_maker.cpp
        src/libalgebra_context_maker.h
        src/libalgebra_context.cpp
        )
target_sources(esig_libalgebra_context
        PRIVATE ${ESIG_LA_GENERATED_FILES})


enable_testing()
add_subdirectory(tests)
