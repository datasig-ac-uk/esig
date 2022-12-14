cmake_minimum_required(VERSION 3.19)

include (GNUInstallDirs)


find_package(Boost CONFIG REQUIRED)


add_library(esig_py_module SHARED get_vector_construction_data.cpp get_vector_construction_data.h py_tensor_key_iterator.cpp py_tensor_key_iterator.h py_lie_key_iterator.cpp py_lie_key_iterator.h)
#pybind11_add_module(esig_py_module MODULE WITH_SOABI)
set_target_properties(esig_py_module PROPERTIES
        LIBRARY_OUTPUT_NAME _esig
        )


target_sources(esig_py_module PRIVATE
        py_esig.cpp
        py_esig.h
        py_algebra.cpp
        py_algebra.h
        py_intervals.cpp
        py_intervals.h
        py_paths.cpp
        py_paths.h
        py_scalars.cpp
        py_scalars.h
        interval.cpp
        interval.h
        dyadic_interval.cpp
        dyadic_interval.h
        dyadic.cpp
        dyadic.h
        real_interval.cpp
        real_interval.h
        recombine.cpp
        recombine.h
        segmentation.cpp
        segmentation.h
        scalar_type.cpp
        scalar_type.h
        lie.cpp
        lie.h
        free_tensor.cpp
        free_tensor.h
        py_fmt_to_esig_fmt.cpp
        py_fmt_to_esig_fmt.h
        py_buffer_to_buffer.cpp
        py_buffer_to_buffer.h
        py_arg_to_ctype.cpp
        py_arg_to_ctype.h
        ctype_to_py_fmt.cpp
        ctype_to_py_fmt.h
        ctype_to_npy_dtype.cpp
        ctype_to_npy_dtype.h
        py_tensor_key.cpp
        py_tensor_key.h
        py_lie_letter.cpp
        py_lie_letter.h
        py_lie_key.cpp
        py_lie_key.h
        kwargs_to_vector_construction.cpp
        kwargs_to_vector_construction.h
        list_to_buffer.cpp
        list_to_buffer.h
        )

target_link_libraries(esig_py_module PRIVATE
        esig_common
        esig_algebra
        esig_paths
        recombine::recombine
        pybind11::pybind11
        )

install(TARGETS esig_py_module
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
