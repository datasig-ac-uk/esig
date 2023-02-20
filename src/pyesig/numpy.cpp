//
// Created by user on 20/02/23.
//

#include "numpy.h"

namespace {

enum npy_dtypes : int {
    NPY_BOOL = 0,
    NPY_BYTE,
    NPY_UBYTE,
    NPY_SHORT,
    NPY_USHORT,
    NPY_INT,
    NPY_UINT,
    NPY_LONG,
    NPY_ULONG,
    NPY_LONGLONG,
    NPY_ULONGLONG,
    NPY_FLOAT,
    NPY_DOUBLE,
    NPY_LONGDOUBLE,
    NPY_CFLOAT,
    NPY_CDOUBLE,
    NPY_CLONGDOUBLE,
    NPY_OBJECT = 17,
    NPY_STRING,
    NPY_UNICODE,
    NPY_VOID,
    /*
                     * New 1.6 types appended, may be integrated
                     * into the above in 2.0.
                     */
    NPY_DATETIME,
    NPY_TIMEDELTA,
    NPY_HALF,

    NPY_NTYPES,
    NPY_NOTYPE,
    NPY_CHAR,
    NPY_USERDEF = 256, /* leave room for characters */

    /* The number of types not including the new 1.6 types */
    NPY_NTYPES_ABI_COMPATIBLE = 21
};

} // namespace

const esig::scalars::scalar_type *esig::python::npy_dtype_to_ctype(py::dtype dtype) {
    const esig::scalars::scalar_type* type = nullptr;

    switch (dtype.num()) {
        case NPY_FLOAT:
            type = esig::scalars::dtl::scalar_type_holder<float>::get_type();
            break;
        case NPY_DOUBLE:
            type = esig::scalars::dtl::scalar_type_holder<double>::get_type();
            break;
        default:
            // Default behaviour, promote to double
            type = esig::scalars::dtl::scalar_type_holder<double>::get_type();
            break;
    }

    return type;
}

py::dtype esig::python::ctype_to_npy_dtype(const esig::scalars::scalar_type *type) {
    if (type == scalars::dtl::scalar_type_holder<double>::get_type()) {
        return py::dtype("d");
    }
    if (type == scalars::dtl::scalar_type_holder<float>::get_type()) {
        return py::dtype("f");
    }

    throw py::type_error("unsupported data type");
}

std::string esig::python::npy_dtype_to_identifier(py::dtype dtype) {
    std::string identifier;

    switch (dtype.num()) {
        case NPY_FLOAT:
            identifier = "f32";
            break;
        case NPY_DOUBLE:
            identifier = "f64";
            break;
        case NPY_INT:
            identifier = "i32";
            break;
        case NPY_UINT:
            identifier = "u32";
            break;
        case NPY_LONG: {
                if (sizeof(long) == sizeof(int)) {
                    identifier = "i32";
                } else {
                    identifier = "i64";
                }
            }
            break;
        case NPY_ULONG: {
                if (sizeof(long) == sizeof(int)) {
                    identifier = "ui32";
                } else {
                    identifier = "u64";
                }
            }
            break;
        case NPY_LONGLONG:
            identifier = "i64";
            break;
        case NPY_ULONGLONG:
            identifier = "u64";
            break;

        case NPY_BOOL:
        case NPY_BYTE:
            identifier = "i8";
            break;
        case NPY_UBYTE:
            identifier = "u8";
            break;
        case NPY_SHORT:
            identifier = "i16";
            break;
        case NPY_USHORT:
            identifier = "u16";
            break;

        default:
            throw py::type_error("unsupported dtype");
    }

    return identifier;
}
