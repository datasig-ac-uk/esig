


if (SKBUILD)
    # This only applies if we're working with skbuild becaues otherwise there
    # is no reason to expect that we have pip installed mkl-devel

#    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)"
#            OUTPUT_VARIABLE MKL_ROOT
#            OUTPUT_STRIP_TRAILING_WHITESPACE
#            )
#    message(STATUS "MKL_ROOT: ${MKL_ROOT}")

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.machine())"
            OUTPUT_VARIABLE ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    message(STATUS "ARCH: ${ARCH}")


#    set(ENV{MKLROOT} ${MKL_ROOT})

    if (ARCH STREQUAL "x86_64" OR ARCH STREQUAL "AMD64")
        set(BLA_VENDOR Intel10_64ilp)
    elseif(ARCH STREQUAL "x86")
        set(BLA_VENDOR Intel10_32)
    else()
        message(FATAL_ERROR "Unrecognised architecture")
    endif()

endif()


find_package(BLAS REQUIRED)

message(STATUS "BLAS: ${BLAS_LIBRARIES}")