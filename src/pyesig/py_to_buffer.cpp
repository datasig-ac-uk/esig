//
// Created by user on 20/02/23.
//

#include "py_to_buffer.h"

#include <string>

#include <boost/endian.hpp>

#include <esig/scalars.h>

#include "numpy.h"
#include "py_fmt_to_esig_fmt.h"

using namespace esig;

namespace {
// Internal helper structs

struct arg_size_info {
    idimn_t num_values;
    idimn_t num_keys;
};

enum class ground_data_type {
    UnSet,
    Scalars,
    KeyValuePairs
};

} // namespace

static char format_to_type_char(const std::string& fmt) {
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



const scalars::scalar_type *python::py_buffer_fmt_to_stype(const std::string &fmt) {


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

const scalars::scalar_type *python::scalar_type_for_pytype(py::handle object) {
    // For now, just return double type. In the future, replace
    // with lookups to sensible choices
    return scalars::dtl::scalar_type_holder<double>::get_type();
}

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



static void check_and_set_dtype(scalars::key_scalar_array &result,
                                python::py_to_buffer_options &options,
                                py::handle arg) {
    if (options.type == nullptr) {
        if (options.no_check_imported) {
            options.type = scalars::dtl::scalar_type_holder<double>::get_type();
        } else {
            options.type = python::scalar_type_for_pytype(arg);
        }
        result = scalars::key_scalar_array(options.type);
    }
}

static bool check_ground_type(py::handle object, ground_data_type& ground_type, python::py_to_buffer_options& options) {
    using ground_data_type::KeyValuePairs;
    using ground_data_type::Scalars;
    using ground_data_type::UnSet;

    py::handle scalar;
    if (is_scalar(object)) {
        if (ground_type == UnSet) {
            ground_type = Scalars;
        } else if (ground_type != Scalars) {
            throw py::value_error("inconsistent scalar/key-scalar-pair data");
        }
        scalar = object;
    } else if (is_kv_pair(object)) {
        if (ground_type == UnSet) {
            ground_type = KeyValuePairs;
        } else if (ground_type != KeyValuePairs) {
            throw py::value_error("inconsistent scalar/key-scalar-pair data");
        }
        scalar = object.cast<py::tuple>()[1];
    } else {
        // TODO: Check non-int/float scalar types
        return false;
    }

    const scalars::scalar_type* this_type;
    if (options.no_check_imported) {
        this_type = scalars::dtl::scalar_type_holder<double>::get_type();
    } else {
        this_type = python::scalar_type_for_pytype(scalar);
    }
    if (options.type == nullptr) {
        options.type = this_type;
    }
    // TODO: Insert check for compatibility if the scalar type is set

    return true;
}


static void compute_size_and_type_recurse(
    python::py_to_buffer_options& options,
    std::vector<py::object>& leaves,
    const py::handle& object,
    ground_data_type& ground_type,
    dimn_t depth)
{
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
        for (auto&& item : sequence) {
            check_ground_type(item, ground_type, options);
        }

        leaves.push_back(std::move(sequence));
    } else if (py::isinstance<py::sequence>(item0)) {
        for (auto&& sibling : sequence) {
            compute_size_and_type_recurse(options, leaves, sibling, ground_type, depth+1);
        }
    } else {
        throw py::type_error("unexpected type in array argument");
    }

}


static arg_size_info compute_size_and_type(
    python::py_to_buffer_options& options,
    std::vector<py::object>& leaves,
    py::handle arg)
{
    arg_size_info info = {1, 0};

    assert(py::isinstance<py::sequence>(arg));

    ground_data_type ground_type = ground_data_type::UnSet;
    compute_size_and_type_recurse(options, leaves, arg, ground_type, 0);

    for (auto& shape_i : options.shape) {
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

static void handle_sequence_element(scalars::scalar_pointer scalar_ptr, key_type* key_ptr, py::handle object) {
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


esig::scalars::key_scalar_array esig::python::py_to_buffer(py::handle arg, py_to_buffer_options& options) {
    scalars::key_scalar_array result(options.type);

    // First handle the single number cases
    if (py::isinstance<py::float_>(arg) || py::isinstance<py::int_>(arg)) {
        if (!options.allow_scalar) {
            throw py::value_error("scalar value not permitted in this context");
        }

        check_and_set_dtype(result, options, arg);
        result.allocate_scalars(1);

        handle_sequence_element(result, nullptr, arg);
    } else if (is_kv_pair(arg)) {
        /*
         * Now for tuples of length 2, which we expect to be a kv-pair
         */
        auto tpl_arg = py::reinterpret_borrow<py::tuple>(arg);

        auto value = tpl_arg[1];
        check_and_set_dtype(result, options, value);
        result.allocate_scalars(1);
        result.allocate_keys(1);

        handle_sequence_element(result, result.keys(), arg);
    } else if (py::isinstance<py::dict>(arg)) {
        auto dict_arg = py::reinterpret_borrow<py::dict>(arg);

        /*
         * With a dict argument we're expecting either
         * dict[int, Scalar] or dict[int, list[Scalar] | buffer
         */
        // TODO: fill this case in
        throw py::type_error("dict to buffer not currently supported");
    } else if (py::isinstance<py::sequence>(arg)) {
        std::vector<py::object> leaves;
        auto size_info = compute_size_and_type(options, leaves, arg);

        result = scalars::key_scalar_array(options.type);
        result.allocate_scalars(size_info.num_values);
        result.allocate_keys(size_info.num_keys);

        scalars::scalar_pointer scalar_ptr(result);
        auto* key_ptr = result.keys();
        for (auto&& leaf : leaves) {
            auto leaf_seq = py::reinterpret_borrow<py::sequence>(leaf);
            for (auto&& object : leaf_seq) {
                handle_sequence_element(scalar_ptr++, key_ptr++, object);
            }
        }
    }

    return result;
}
