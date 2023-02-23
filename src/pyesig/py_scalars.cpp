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
#include "scalar_meta.h"


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
            format = type_id_of<double>();
            break;
        case 'f':
            format = type_id_of<float>();
            break;
        case 'l': {
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<int>();
            } else {
                format = type_id_of<long long>();
            }
            break;
        }
        case 'q':
            format = scalars::type_id_of<long long>();
            break;
        case 'L':
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<unsigned int>();
            } else {
                format = type_id_of<unsigned long long>();
            }
            break;
        case 'Q':
            format = type_id_of<unsigned long long>();
            break;
        case 'i':
            format = type_id_of<int>();
            break;
        case 'I':
            format = type_id_of<unsigned int>();
            break;
        case 'n':
            format = type_id_of<scalars::signed_size_type_marker>();
            break;
        case 'N':
            format = type_id_of<scalars::unsigned_size_type_marker>();
            break;
        case 'h':
            format = type_id_of<short>();
            break;
        case 'H':
            format = type_id_of<unsigned short>();
            break;
        case 'b':
        case 'c':
            format = type_id_of<char>();
            break;
        case 'B':
            format = type_id_of<unsigned char>();
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
        default:
            throw py::type_error("no matching type for buffer type " + std::string(&python_format));
    }
    // TODO: Add custom type handling

    return scalar_type::of<double>();
}

const scalars::scalar_type *python::py_arg_to_ctype(const py::object &object) {
    if (py::isinstance(object, python::get_scalar_metaclass())) {
        return reinterpret_cast<python::PyScalarMetaType*>(object.ptr())->tp_ctype;
    }
    return nullptr;
}

const scalars::scalar_type *python::py_type_to_scalar_type(const py::type &type) {
    if (Py_IS_TYPE(type.ptr(), &PyFloat_Type)) {
        return scalars::scalar_type::of<double>();
    } else if (Py_IS_TYPE(type.ptr(), &PyLong_Type)) {
        return scalars::scalar_type::of<double>();
    }



    throw py::type_error("no matching scalar type for type " + pytype_name(type));
}
py::type python::scalar_type_to_py_type(const scalars::scalar_type *type) {

    if (type == scalars::scalar_type::of<float>() || type == scalars::scalar_type::of<double>()) {
        return py::reinterpret_borrow<py::type>((PyObject*) &PyFloat_Type);
    }


    throw py::type_error("no matching type for type " + type->info().name);
}


static const scalars::scalar_type* dlpack_dtype_to_scalar_type(DLDataType dtype, DLDevice device) {
    return scalars::scalar_type::from_type_details({dtype.code, dtype.bits, dtype.lanes,
                                                    {device.device_type, device.device_id}});
}

static inline void dl_copy_strided(std::int32_t ndim,
                            std::int64_t* shape,
                            std::int64_t* strides,
                            scalars::scalar_pointer src,
                            scalars::scalar_pointer dst
)
{
    if (ndim == 1) {
        if (strides[0] == 1) {
            dst.type()->convert_copy(dst.ptr(), src, shape[0]);
        } else {
            for (std::int64_t i=0; i<shape[0]; ++i) {
                dst[i] = src[i*strides[0]];
            }
        }
    } else {
        auto* next_shape = shape+1;
        auto* next_stride = strides+1;

        for (std::int64_t j=0; j<shape[0]; ++j) {
            dl_copy_strided(ndim - 1, next_shape, next_stride, src + j*strides[0], dst + j*shape[0]);
        }
    }

}


static bool try_fill_buffer_dlpack(scalars::key_scalar_array& buffer,
                                   python::py_to_buffer_options& options,
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
        if (tensor == nullptr || tensor->dl_tensor.data == nullptr || tensor->dl_tensor.shape == nullptr) {
            return false;
        }

        auto& dltensor = tensor->dl_tensor;
        auto* data = reinterpret_cast<char*>(dltensor.data) + dltensor.byte_offset*(dltensor.dtype.bits >> 3);
        auto ndim = dltensor.ndim;
        auto* shape = dltensor.shape;
        auto* strides = dltensor.strides;

        const scalars::scalar_type* tensor_stype = nullptr;

        try {
            tensor_stype = dlpack_dtype_to_scalar_type(dltensor.dtype, dltensor.device);
        } catch (...) {
            // TODO: handle this case more gracefully
            return false;
        }

        if (options.type == nullptr) {
            options.type = tensor_stype;
            buffer = scalars::key_scalar_array(options.type);
        }

        idimn_t size = 1;
        for (auto i=0; i<ndim; ++i) {
            size *= static_cast<idimn_t>(shape[i]);
        }

        if (strides == nullptr) {
            buffer = scalars::scalar_array(data, options.type, size);
        } else {
            buffer.allocate_scalars(size);
            scalars::scalar_pointer p(data, tensor_stype);
            dl_copy_strided(ndim, shape, strides, p, buffer + 0);

        }

        if (tensor->deleter != nullptr) {
            options.cleanup = [tensor]() {
              tensor->deleter(tensor);
            };
        }

        return true;
    }



    return false;
}

struct arg_size_info {
    idimn_t num_values;
    idimn_t num_keys;
};

enum class ground_data_type {
    UnSet,
    Scalars,
    KeyValuePairs
};

static bool is_scalar(py::handle arg) {
    return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg);
}

static bool is_kv_pair(py::handle arg) {
    if (py::isinstance<py::tuple>(arg)) {
        auto tpl = py::reinterpret_borrow<py::tuple>(arg);
        return (tpl.size() == 2 && py::isinstance<py::int_>(tpl[0]));
    }
    return false;
}

static bool check_ground_type(py::handle object, ground_data_type &ground_type, python::py_to_buffer_options &options) {
    py::handle scalar;
    if (is_scalar(object)) {
        if (ground_type == ground_data_type::UnSet) {
            ground_type = ground_data_type::Scalars;
        } else if (ground_type != ground_data_type::Scalars) {
            throw py::value_error("inconsistent scalar/key-scalar-pair data");
        }
        scalar = object;
    } else if (is_kv_pair(object)) {
        if (ground_type == ground_data_type::UnSet) {
            ground_type = ground_data_type::KeyValuePairs;
        } else if (ground_type != ground_data_type::KeyValuePairs) {
            throw py::value_error("inconsistent scalar/key-scalar-pair data");
        }
        scalar = object.cast<py::tuple>()[1];
    } else {
        // TODO: Check non-int/float scalar types
        return false;
    }

    const scalars::scalar_type *this_type;
    if (options.no_check_imported) {
        this_type = scalars::dtl::scalar_type_holder<double>::get_type();
    } else {
        this_type = python::py_type_to_scalar_type(py::reinterpret_borrow<py::type>(scalar.get_type()));
    }
    if (options.type == nullptr) {
        options.type = this_type;
    }
    // TODO: Insert check for compatibility if the scalar type is set

    return true;
}

static void compute_size_and_type_recurse(
    python::py_to_buffer_options &options,
    std::vector<py::object> &leaves,
    const py::handle &object,
    ground_data_type &ground_type,
    dimn_t depth) {
    if (!py::isinstance<py::sequence>(object)) {
        throw py::type_error("unexpected type in array argument");
    }
    auto sequence = py::reinterpret_borrow<py::sequence>(object);
    auto length = static_cast<idimn_t>(py::len(sequence));

    if (options.shape.size() == depth) {
        // We've not visited this depth before,
        // add our length to the list
        options.shape.push_back(length);
    } else {
        // We have visited this depth before,
        // check our length is consistent with the others
        if (length != options.shape[depth]) {
            throw py::value_error("ragged arrays are not supported");
        }
    }
    if (length == 0) {
        // if the length is zero, there is nothing left to do
        return;
    }

    /*
     * Now we handle the meat of the recursion.
     * If we find scalars in side this level, then we're
     * at the bottom, and we stop recursing. Otherwise, we
     * find another layer of nested sequences and we
     * recurse into those.
     */

    auto item0 = sequence[0];
    if (check_ground_type(item0, ground_type, options)) {
        // We've hit the bottom, container holds either
        // scalars or key-scalar pairs.

        // Check all the scalar types are the same
        for (auto &&item : sequence) {
            check_ground_type(item, ground_type, options);
        }

        leaves.push_back(std::move(sequence));
    } else if (py::isinstance<py::sequence>(item0)) {
        for (auto &&sibling : sequence) {
            compute_size_and_type_recurse(options, leaves, sibling, ground_type, depth + 1);
        }
    } else {
        throw py::type_error("unexpected type in array argument");
    }
}

static arg_size_info compute_size_and_type(
    python::py_to_buffer_options &options,
    std::vector<py::object> &leaves,
    py::handle arg) {
    arg_size_info info = {1, 0};

    assert(py::isinstance<py::sequence>(arg));

    ground_data_type ground_type = ground_data_type::UnSet;
    compute_size_and_type_recurse(options, leaves, arg, ground_type, 0);

    for (auto &shape_i : options.shape) {
        info.num_values *= shape_i;
    }
    if (info.num_values == 0 || ground_type == ground_data_type::UnSet) {
        options.shape.clear();
        leaves.clear();
    } else if (ground_type == ground_data_type::KeyValuePairs) {
        info.num_keys = info.num_values;
    }

    return info;
}

void python::assign_py_object_to_scalar(scalars::scalar_pointer p, py::handle object) {

    if (py::isinstance<py::float_>(object)) {
        *p = object.cast<double>();
    } else if (py::isinstance<py::int_>(object)) {
        *p = object.cast<long long>();
    } else {
        // TODO: other checks

        auto tp = py::type::of(object);
        throw py::value_error("bad conversion from " + tp.cast<std::string>() + " to " + p.type()->info().name);
    }
}

static void handle_sequence_element(scalars::scalar_pointer scalar_ptr, key_type *key_ptr, py::handle object) {
    if (key_ptr != nullptr) {
        // Expecting key-value tuplesl
        auto tpl = py::reinterpret_borrow<py::tuple>(object);
        *key_ptr = tpl[0].cast<key_type>();
        python::assign_py_object_to_scalar(scalar_ptr, tpl[1]);
    } else {
        // Expecting scalars
        python::assign_py_object_to_scalar(scalar_ptr, object);
    }
}

static void check_and_set_dtype(scalars::key_scalar_array &result,
                                python::py_to_buffer_options &options,
                                const py::object& arg) {
    if (options.type == nullptr) {
        if (options.no_check_imported) {
            options.type = scalars::dtl::scalar_type_holder<double>::get_type();
        } else {
            options.type = python::py_type_to_scalar_type(arg);
        }
        result = scalars::key_scalar_array(options.type);
    }
}

scalars::key_scalar_array python::py_to_buffer(const py::object &object, python::py_to_buffer_options &options) {
    scalars::key_scalar_array result(options.type);

    // First handle the single number cases
    if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object)) {
        if (!options.allow_scalar) {
            throw py::value_error("scalar value not permitted in this context");
        }

        check_and_set_dtype(result, options, object);
        result.allocate_scalars(1);

        handle_sequence_element(result, nullptr, object);
    } else if (is_kv_pair(object)) {
        /*
         * Now for tuples of length 2, which we expect to be a kv-pair
         */
        auto tpl_arg = py::reinterpret_borrow<py::tuple>(object);

        auto value = tpl_arg[1];
        check_and_set_dtype(result, options, value);
        result.allocate_scalars(1);
        result.allocate_keys(1);

        handle_sequence_element(result, result.keys(), object);
    } else if (try_fill_buffer_dlpack(result, options, object)) {
        // If we used the dlpack interface, then the result is
        // already constructed.
    } else if (py::isinstance<py::buffer>(object)) {
        // Fall back to the buffer protocol
        auto info = py::reinterpret_borrow<py::buffer>(object).request();
        auto type_id = py_buffer_to_type_id(info);

        if (options.type == nullptr) {
            options.type = scalars::scalar_type::for_id(type_id);
            result = scalars::key_scalar_array(options.type);
        }

        result.allocate_scalars(info.size);
        options.type->convert_copy(result, info.ptr, static_cast<dimn_t>(info.size), type_id);
    } else if (py::isinstance<py::dict>(object)) {
        auto dict_arg = py::reinterpret_borrow<py::dict>(object);

        /*
         * With a dict argument we're expecting either
         * dict[int, Scalar] or dict[int, list[Scalar] | buffer
         */
        // TODO: fill this case in
        throw py::type_error("dict to buffer not currently supported");
    } else if (py::isinstance<py::sequence>(object)) {
        std::vector<py::object> leaves;
        auto size_info = compute_size_and_type(options, leaves, object);

        result = scalars::key_scalar_array(options.type);
        result.allocate_scalars(size_info.num_values);
        result.allocate_keys(size_info.num_keys);

        scalars::scalar_pointer scalar_ptr(result);
        auto *key_ptr = result.keys();
        for (auto &&leaf : leaves) {
            auto leaf_seq = py::reinterpret_borrow<py::sequence>(leaf);
            for (auto &&obj : leaf_seq) {
                handle_sequence_element(scalar_ptr++, key_ptr++, obj);
            }
        }
    }

    return result;
}






void esig::python::init_scalars(py::module_ &m) {
    using namespace esig::scalars;

    py::options options;
    options.disable_function_signatures();

    init_scalar_type(m);

    py::class_<scalar> klass(m, "Scalar", SCALAR_DOC);



}
