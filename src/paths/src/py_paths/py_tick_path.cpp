//
// Created by user on 25/05/22.
//

#include "py_tick_path.h"
#include <esig/paths/python_interface.h>
#include "python_arguments.h"
#include <pybind11/pytypes.h>

using esig::paths::path;
using esig::deg_t;
using esig::dimn_t;
using esig::param_t;
using esig::key_type;
using esig::algebra::vector_type;
using esig::algebra::coefficient_type;

namespace py = pybind11;
using namespace pybind11::literals;

static const char* TICK_PATH_DOC = R"epdoc(Path where increments occur at discrete tick events.
)epdoc";

void esig::paths::init_tick_path(pybind11::module_ &m)
{
    py::class_<esig::paths::tick_path, esig::paths::path_interface> klass(m, "TickPath", TICK_PATH_DOC);


    esig::paths::register_pypath_constructor(
            py::type::of<esig::paths::tick_path>(),
            construct_tick_path);


}


namespace {

template <coefficient_type CType>
struct buffer_constructor_helper
{
    using scalar_type = esig::algebra::type_of_coeff<CType>;

    std::vector<std::pair<key_type, scalar_type>> tmp_buffer;


    void operator()(const py::handle& obj)
    {
        auto key = obj[py::int_(0)].cast<key_type>();
        auto scal = obj[py::int_(1)];

        if (py::isinstance<py::float_>(scal)) {
            tmp_buffer.emplace_back(key, scalar_type(scal.cast<double>()));
        } else {
            tmp_buffer.emplace_back(key, scalar_type(scal.cast<double>()));
        }
    }

    esig::algebra::allocating_data_buffer
    to_buffer(const esig::algebra::context& ctx)
    {
        esig::algebra::allocating_data_buffer result(ctx.pair_alloc(), tmp_buffer.size());
        std::uninitialized_copy(
                tmp_buffer.begin(),
                tmp_buffer.end(),
                reinterpret_cast<std::pair<key_type, scalar_type>*>(result.begin())
        );
        return result;
    }



};

template <coefficient_type CType>
path construct_path_impl(const py::args& args, const py::kwargs& kwargs)
{


    esig::paths::path_metadata md {
            kwargs["width"].cast<deg_t>(),
            kwargs["depth"].cast<deg_t>(),
            kwargs["domain"].cast<esig::real_interval>(),
            CType,
            vector_type::sparse,
            kwargs["result_vec_type"].cast<vector_type>()
    };

    auto ctx = esig::algebra::get_context(md.width, md.depth, CType, {});

    buffer_constructor_helper<CType> helper;
    std::vector<std::pair<param_t, dimn_t>> index;

    for (auto arg : args) {
        for (auto item : arg) {
            auto idx = item[py::int_(0)].cast<param_t>();
            auto data = item[py::int_(1)];
            if (py::isinstance<py::tuple>(data)) {
                helper(data);
                index.emplace_back(idx, 1);
            } else if (py::isinstance<py::iterable>(data)) {
                dimn_t count = 0;
                for (auto val : data) {
                    helper(val);
                    ++count;
                }
                index.emplace_back(idx, count);
            }
        }
    }

    return path(esig::paths::tick_path(
            std::move(md),
            std::move(index),
            helper.to_buffer(*ctx))
    );
}

dimn_t get_number_of_elts(py::object &arg) {
    if (py::isinstance<py::float_>(arg)) {
        return 1;
    }
    if (py::isinstance<py::buffer>(arg)) {
        auto info = arg.cast<py::buffer>().request();
        return info.size;
    }
    if (py::isinstance<py::sequence>(arg)) {
        return py::len(arg);
    }
    if (py::isinstance<py::iterable>(arg)) {
        arg = py::list(arg);
        return py::len(arg);
    }
    try {
        arg = py::float_(arg);
        return 1;
    } catch (...) {
        throw py::value_error("expected float, array, sequence, or iterable");
    }
}


template <coefficient_type CType>
void process_tick_data_impl(esig::algebra::data_buffer& buffer,
                            const esig::paths::path_metadata& md,
                            std::vector<std::pair<py::object, dimn_t>>&& data)
{
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    auto* ptr = buffer.begin();
    for (auto& item : data) {
        if (item.second == 1) {
            esig::paths::copy_pyobject_to_memory<scalar_type>(ptr, item.first);
            ptr += sizeof(scalar_type);
        } else if (py::isinstance<py::buffer>(item.first)) {
            auto info = item.first.cast<py::buffer>().request();
            esig::paths::copy_py_buffer_to_data_buffer<CType>(ptr, info);
            ptr += info.size*sizeof(scalar_type);
        } else if (py::isinstance<py::sequence>(item.first)) {
            // Iterator types should have been converted to lists
            for (auto obj : item.first.cast<py::sequence>()) {
                esig::paths::copy_pyobject_to_memory<scalar_type>(ptr, obj);
                ptr += sizeof(scalar_type);
            }
        } else {
            throw py::value_error("could not convert object to scalar type");
        }
    }
}


void process_tick_data(esig::algebra::data_buffer& buffer,
                       const esig::paths::path_metadata& md,
                       std::vector<std::pair<py::object, dimn_t>>&& data)
{
#define ESIG_SWITCH_FN(CTYPE) process_tick_data_impl<CTYPE>(buffer, md, std::move(data))
    ESIG_MAKE_CTYPE_SWITCH(md.ctype)
#undef ESIG_SWITCH_FN
}

} // namespace



path esig::paths::construct_tick_path(const pybind11::args& args, const pybind11::kwargs &kwargs)
{
    if (args.empty()) {
        throw std::invalid_argument("At least one argument must be provided");
    }

    esig::paths::additional_args additional;
    auto md = esig::paths::parse_kwargs_to_metadata(kwargs, additional);

    std::vector<std::pair<py::object, dimn_t>> data;
    std::vector<std::pair<param_t, dimn_t>> index_data;

    dimn_t count = 0;
    auto process_tuple = [&count, &data, &index_data] (py::object arg) {
        auto pair = arg.cast<std::pair<param_t, py::object>>();
        index_data.emplace_back(pair.first, count);
        auto num = get_number_of_elts(pair.second);
        count += num;
        data.emplace_back(pair.second, num);
    };

    for (auto arg : args) {
        if (py::isinstance<py::tuple>(arg)) {
            process_tuple(arg.cast<py::object>());
        } else if (py::isinstance<py::iterable>(arg)) {
            auto len = len_hint(arg);

            index_data.reserve(index_data.size() + len);
            data.reserve(data.size() + len);
            for (auto in_seq : arg) {
                process_tuple(arg.cast<py::object>());
            }
        } else {
            throw py::type_error("Expected tuple or iterable of tuples");
        }
    }

    algebra::allocating_data_buffer buffer(md.ctx->coefficient_alloc(), count);
    process_tick_data(buffer, md, std::move(data));

    return path(tick_path(std::move(md), std::move(index_data), std::move(buffer)));
}
