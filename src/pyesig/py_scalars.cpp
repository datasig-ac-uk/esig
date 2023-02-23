//
// Created by user on 08/12/22.
//

#include "py_scalars.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <dlpack/dlpack.h>

#include "scalar_type.h"


using namespace esig;

static const char* SCALAR_DOC = R"edoc(
A generic scalar value.
)edoc";

char python::format_to_type_char(const std::string& fmt) {

    char python_format = 0;
    for (const auto &chr : fmt) {
        switch (chr) {
            case '<':// little-endian
#if BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_BIG_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#else
                break;
#endif
            case '>':// big-endian
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#else
                break;
#endif
            case '@':// native
            case '=':// native
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                break;
#endif
            case '!':// network ( = big-endian )
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#else
                break;
#endif
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                break;
            default:
                python_format = chr;
                goto after_loop;
        }
    }
after_loop:
    return python_format;
}

std::string python::py_buffer_to_type_id(const py::buffer_info &info) {
    using scalars::type_id_of;

    auto python_format = format_to_type_char(info.format);
    std::string format;
    switch (python_format) {
        case 'd':
            format = type_id_of<double>::get_id();
            break;
        case 'f':
            format = type_id_of<float>::get_id();
            break;
        case 'l': {
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<int>::get_id();
            } else {
                format = type_id_of<long long>::get_id();
            }
            break;
        }
        case 'q':
            format = scalars::type_id_of<long long>::get_id();
            break;
        case 'L':
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<unsigned int>::get_id();
            } else {
                format = type_id_of<unsigned long long>::get_id();
            }
            break;
        case 'Q':
            format = type_id_of<unsigned long long>::get_id();
            break;
        case 'i':
            format = type_id_of<int>;
            break;
        case 'I':
            format = type_id_of<unsigned int>::get_id();
            break;
        case 'n':
            format = type_id_of<scalars::signed_size_type_marker>::get_id();
            break;
        case 'N':
            format = type_id_of<scalars::unsigned_size_type_marker>::get_id();
            break;
        case 'h':
            format = type_id_of<short>::get_id();
            break;
        case 'H':
            format = type_id_of<unsigned short>::get_id();
            break;
        case 'b':
        case 'c':
            format = type_id_of<char>::get_id();
            break;
        case 'B':
            format = type_id_of<unsigned char>::get_id();
            break;
        default:
            throw std::runtime_error("Unrecognised data format");
    }

    return format;
}

static inline std::string pytype_name(const py::type& type) {
    return std::string(reinterpret_cast<PyTypeObject*>(type.ptr())->tp_name);
}


const scalars::scalar_type *python::py_buffer_to_scalar_type(const py::buffer_info &info) {
    using scalars::scalar_type;

    auto python_format = format_to_type_char(info.format);

    switch (python_format) {
        case 'f':
            return scalar_type::of<float>();
        case 'd':
            return scalar_type::of<double>();
    }
    // TODO: Add custom type handling

    return scalar_type::of<double>();

}



const scalars::scalar_type *python::py_type_to_scalar_type(const py::type &type) {




}
py::type python::scalar_type_to_py_type(const scalars::scalar_type *) {
    return py::type(pybind11::handle(), false);
}

static bool try_fill_buffer_dlpack(scalars::key_scalar_array& buffer,
                                   const py::object& object)
{
    if (hasattr(object, "__dlpack__")) {

        py::capsule dlpack;
        try {
            dlpack = object.attr("__dlpack__")(py::none());
        } catch (...) {
            return false;
        }

        auto* tensor = reinterpret_cast<DLManagedTensor*>(dlpack.get_pointer());

    }

    return false;
}


scalars::key_scalar_array python::py_to_buffer(const py::object &object, python::py_to_buffer_options &options) {
    return scalars::key_scalar_array();
}






void esig::python::init_scalars(py::module_ &m) {
    using namespace esig::scalars;

    py::options options;
    options.disable_function_signatures();

    init_scalar_type(m);

    py::class_<scalar> klass(m, "Scalar", SCALAR_DOC);



}
