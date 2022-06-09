//
// Created by user on 25/05/22.
//

#include "py_tick_path.h"
#include <esig/paths/python_interface.h>
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
    to_buffer(esig::algebra::context& ctx)
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
    if (args.empty()) {
        throw std::invalid_argument("At least one argument must be provided");
    }

    esig::paths::path_metadata md {
            kwargs["width"].cast<deg_t>(),
            kwargs["depth"].cast<deg_t>(),
            kwargs["domain"].cast<esig::real_interval>(),
            CType,
            vector_type::sparse,
            kwargs["result_vec_type"].cast<vector_type>()
    };

    auto ctx = esig::algebra::get_context(md.width, md.depth, CType);

    buffer_constructor_helper<CType> helper;
    std::vector<std::pair<param_t, dimn_t>> index;

    for (auto arg : args) {
        for (auto item : arg) {
            auto idx = item[py::int_(0)].cast<param_t>();
            auto data = item[py::int_(1)];
            if (py::isinstance<py::iterable>(data)) {
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


} // namespace


path esig::paths::construct_tick_path(const pybind11::args& args, const pybind11::kwargs &kwargs)
{
#define ESIG_SWITCH_FN(CTYPE) construct_path_impl<CTYPE>(args, kwargs)
    ESIG_MAKE_CTYPE_SWITCH(kwargs["coeff_type"].cast<coefficient_type>())
#undef ESIG_SWITCH_FN
}
