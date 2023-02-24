//
// Created by user on 08/12/22.
//

#include "py_scalars.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>

#include "scalar_meta.h"
#include "scalar_type.h"

using namespace esig;

static const char *SCALAR_DOC = R"edoc(
A generic scalar value.
)edoc";

char python::format_to_type_char(const std::string &fmt) {

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

static inline std::string pytype_name(const py::type &type) {
    return std::string(reinterpret_cast<PyTypeObject *>(type.ptr())->tp_name);
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
        return reinterpret_cast<python::PyScalarMetaType *>(object.ptr())->tp_ctype;
    }
    if (py::isinstance<py::str>(object)) {
        return scalars::scalar_type::for_id(object.cast<std::string>());
    }
    return nullptr;
}

const scalars::scalar_type *python::py_type_to_scalar_type(const py::type &type) {
    if (Py_Is(type.ptr(), (PyObject *)&PyFloat_Type)) {
        return scalars::scalar_type::of<double>();
    } else if (Py_Is(type.ptr(), (PyObject *)&PyLong_Type)) {
        return scalars::scalar_type::of<double>();
    }

    throw py::type_error("no matching scalar type for type " + pytype_name(type));
}
py::type python::scalar_type_to_py_type(const scalars::scalar_type *type) {

    if (type == scalars::scalar_type::of<float>() || type == scalars::scalar_type::of<double>()) {
        return py::reinterpret_borrow<py::type>((PyObject *)&PyFloat_Type);
    }

    throw py::type_error("no matching type for type " + type->info().name);
}

static const scalars::scalar_type *dlpack_dtype_to_scalar_type(DLDataType dtype, DLDevice device) {
    return scalars::scalar_type::from_type_details({dtype.code, dtype.bits, dtype.lanes, {device.device_type, device.device_id}});
}

static inline void dl_copy_strided(std::int32_t ndim,
                                   std::int64_t *shape,
                                   std::int64_t *strides,
                                   scalars::scalar_pointer src,
                                   scalars::scalar_pointer dst) {
    if (ndim == 1) {
        if (strides[0] == 1) {
            dst.type()->convert_copy(dst.ptr(), src, shape[0]);
        } else {
            for (std::int64_t i = 0; i < shape[0]; ++i) {
                dst[i] = src[i * strides[0]];
            }
        }
    } else {
        auto *next_shape = shape + 1;
        auto *next_stride = strides + 1;

        for (std::int64_t j = 0; j < shape[0]; ++j) {
            dl_copy_strided(ndim - 1, next_shape, next_stride, src + j * strides[0], dst + j * shape[0]);
        }
    }
}

static inline void buffer_copy_strided(py::ssize_t ndim,
                                       py::ssize_t *shape,
                                       py::ssize_t *strides,
                                       scalars::scalar_pointer src,
                                       scalars::scalar_pointer dst) {
}

static inline void update_dtype_and_allocate(scalars::key_scalar_array &result, python::py_to_buffer_options &options, idimn_t no_values, idimn_t no_keys) {

    if (options.type != nullptr) {
        result = scalars::key_scalar_array(options.type);
        result.allocate_scalars(no_values);
        result.allocate_keys(no_keys);
    } else if (no_values > 0) {
        throw py::type_error("unable to deduce a suitable scalar type");
    }
}

static bool try_fill_buffer_dlpack(scalars::key_scalar_array &buffer,
                                   python::py_to_buffer_options &options,
                                   const py::object &object) {
    py::capsule dlpack;
    dlpack = object.attr("__dlpack__")(py::none());

    auto *tensor = reinterpret_cast<DLManagedTensor *>(dlpack.get_pointer());
    if (tensor == nullptr) {
        throw py::value_error("__dlpack__ returned invalid object");
    }

    auto &dltensor = tensor->dl_tensor;
    auto *data = reinterpret_cast<char *>(dltensor.data);
    auto ndim = dltensor.ndim;
    auto *shape = dltensor.shape;
    auto *strides = dltensor.strides;


    // This function throws if no matching dtype is found
    const auto *tensor_stype = dlpack_dtype_to_scalar_type(dltensor.dtype, dltensor.device);
    if (options.type == nullptr) {
        options.type = tensor_stype;
        buffer = scalars::key_scalar_array(options.type);
    }

    if (data == nullptr) {
        // The array is empty, empty result.
        return true;
    }

    idimn_t size = 1;
    for (auto i = 0; i < ndim; ++i) {
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

struct arg_size_info {
    idimn_t num_values;
    idimn_t num_keys;
};

enum class ground_data_type {
    UnSet,
    Scalars,
    KeyValuePairs
};

static inline bool is_scalar(py::handle arg) {
    return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg);
}

static inline bool is_key(py::handle arg, python::alternative_key_type *alternative) {
    if (alternative != nullptr) {
        return py::isinstance<py::int_>(arg) || py::isinstance(arg, alternative->py_key_type);
    } else if (py::isinstance<py::int_>(arg)) {
        return true;
    }
}

static inline bool is_kv_pair(py::handle arg, python::alternative_key_type *alternative) {
    if (py::isinstance<py::tuple>(arg)) {
        auto tpl = py::reinterpret_borrow<py::tuple>(arg);
        if (tpl.size() == 2) {
            return is_key(tpl[0], alternative);
        }
    }
    return false;
}

static void check_and_set_dtype(python::py_to_buffer_options &options,
                                py::handle arg) {
    if (options.type == nullptr) {
        if (options.no_check_imported) {
            options.type = scalars::dtl::scalar_type_holder<double>::get_type();
        } else {
            options.type = python::py_type_to_scalar_type(py::type::of(arg));
        }
    }
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
    } else if (is_kv_pair(object, options.alternative_key)) {
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

    check_and_set_dtype(options, scalar);

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
    if (depth > options.max_nested) {
        throw py::value_error("maximum nested array limit reached in this context");
    }

    auto sequence = py::reinterpret_borrow<py::sequence>(object);
    auto length = static_cast<idimn_t>(py::len(sequence));

    if (options.shape.size() == depth) {
        // We've not visited this depth before,
        // add our length to the list
        options.shape.push_back(length);
    } else if (ground_type == ground_data_type::Scalars) {
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
    } else if (py::isinstance<py::dict>(item0)) {
        auto dict = py::reinterpret_borrow<py::dict>(item0);

        if (depth == options.max_nested) {
            throw py::value_error("maximum nested depth reached in this context");
        }
        switch (ground_type) {
            case ground_data_type::UnSet:
                ground_type = ground_data_type::KeyValuePairs;
            case ground_data_type::KeyValuePairs:
                break;
            default:
                throw py::type_error("mismatched types in array argument");
        }

        if (!dict.empty()) {
            auto kv = *dict.begin();
            check_and_set_dtype(options, kv.second);
        }

        leaves.push_back(dict);

    } else {
        throw py::type_error("unexpected type in array argument");
    }
}

static arg_size_info compute_size_and_type(
    python::py_to_buffer_options &options,
    std::vector<py::object> &leaves,
    py::handle arg) {
    arg_size_info info = {0, 0};

    assert(py::isinstance<py::sequence>(arg));

    ground_data_type ground_type = ground_data_type::UnSet;
    compute_size_and_type_recurse(options, leaves, arg, ground_type, 0);

    if (ground_type == ground_data_type::KeyValuePairs) {
        options.shape.clear();

        for (const auto &obj : leaves) {
            auto size = static_cast<idimn_t>(py::len(obj));

            options.shape.push_back(size);
            info.num_values += size;
            info.num_keys += size;
        }
    } else {
        info.num_values = 1;
        for (auto &shape_i : options.shape) {
            info.num_values *= shape_i;
        }
    }

    if (info.num_values == 0 || ground_type == ground_data_type::UnSet) {
        options.shape.clear();
        leaves.clear();
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

static void handle_sequence_tuple(scalars::scalar_pointer scalar_ptr, key_type *key_ptr, py::handle tpl_o, python::py_to_buffer_options &options) {
    auto tpl = py::reinterpret_borrow<py::tuple>(tpl_o);
    auto key = tpl[0];
    if (options.alternative_key != nullptr && py::isinstance(key, options.alternative_key->py_key_type)) {
        *key_ptr = options.alternative_key->converter(key);
    } else {
        *key_ptr = key.cast<key_type>();
    }

    python::assign_py_object_to_scalar(scalar_ptr, tpl[1]);
}

static void handle_dict(scalars::scalar_pointer &scalar_ptr,
                        key_type *&key_ptr,
                        python::py_to_buffer_options &options,
                        py::handle dict_o) {

    for (auto obj : py::reinterpret_borrow<py::dict>(dict_o)) {
        // dict iterator yields pairs [key, obj]
        // Expecting key-value tuplesl
        auto key = obj.first;
        if (options.alternative_key != nullptr && py::isinstance(key, options.alternative_key->py_key_type)) {
            *(key_ptr++) = options.alternative_key->converter(key);
        } else {
            *(key_ptr++) = key.cast<key_type>();
        }

        python::assign_py_object_to_scalar(scalar_ptr++, obj.second);
    }
}

scalars::key_scalar_array python::py_to_buffer(const py::object &object, python::py_to_buffer_options &options) {
    scalars::key_scalar_array result(options.type);

    // First handle the single number cases
    if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object)) {
        if (!options.allow_scalar) {
            throw py::value_error("scalar value not permitted in this context");
        }

        check_and_set_dtype(options, object);
        update_dtype_and_allocate(result, options, 1, 0);

        assign_py_object_to_scalar(result, object);
    } else if (is_kv_pair(object, options.alternative_key)) {
        /*
         * Now for tuples of length 2, which we expect to be a kv-pair
         */
        auto tpl_arg = py::reinterpret_borrow<py::tuple>(object);

        auto value = tpl_arg[1];
        check_and_set_dtype(options, value);
        update_dtype_and_allocate(result, options, 1, 1);

        handle_sequence_tuple(result, result.keys(), object, options);
    } else if (py::hasattr(object, "__dlpack__")) {
        // If we used the dlpack interface, then the result is
        // already constructed.
        try_fill_buffer_dlpack(result, options, object);

    } else if (py::isinstance<py::buffer>(object)) {
        // Fall back to the buffer protocol
        auto info = py::reinterpret_borrow<py::buffer>(object).request();
        auto type_id = py_buffer_to_type_id(info);

        if (options.type == nullptr) {
            options.type = scalars::scalar_type::for_id(type_id);
            result = scalars::key_scalar_array(options.type);
        }

        update_dtype_and_allocate(result, options, info.size, 0);

        // The only way type can still be null is if there are no elements.
        if (options.type != nullptr) {
            options.type->convert_copy(result, info.ptr, static_cast<dimn_t>(info.size), type_id);
            options.shape.assign(info.shape.begin(), info.shape.end());
        }


    } else if (py::isinstance<py::dict>(object)) {
        auto dict_arg = py::reinterpret_borrow<py::dict>(object);
        options.shape.push_back(static_cast<idimn_t>(dict_arg.size()));

        if (!dict_arg.empty()) {
            auto kv = *dict_arg.begin();
            check_and_set_dtype(options, kv.second);

            update_dtype_and_allocate(result, options, options.shape[0], options.shape[0]);

            scalars::scalar_pointer ptr(result);
            key_type *key_ptr = result.keys();

            handle_dict(ptr, key_ptr, options, dict_arg);
        }
    } else if (py::isinstance<py::sequence>(object)) {
        std::vector<py::object> leaves;
        auto size_info = compute_size_and_type(options, leaves, object);

        update_dtype_and_allocate(result, options, size_info.num_values, size_info.num_keys);

        scalars::scalar_pointer scalar_ptr(result);

        if (size_info.num_keys == 0) {
            // Scalar info only.
            for (auto leaf : leaves) {
                auto leaf_seq = py::reinterpret_borrow<py::sequence>(leaf);
                for (auto obj : leaf_seq) {
                    assign_py_object_to_scalar(scalar_ptr++, obj);
                }
            }
        } else {
            auto *key_ptr = result.keys();
            assert(size_info.num_values == 0 || key_ptr != nullptr);
            for (auto leaf : leaves) {
                // Key-value
                if (py::isinstance<py::dict>(leaf)) {
                    handle_dict(scalar_ptr, key_ptr, options, leaf);
                } else {
                    for (auto obj : py::reinterpret_borrow<py::sequence>(leaf)) {
                        handle_sequence_tuple(scalar_ptr++, key_ptr++, obj, options);
                    }
                }
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
