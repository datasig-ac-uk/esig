//
// Created by user on 22/07/22.
//

#include "py_piecewise_lie_path.h"
#include <esig/paths/python_interface.h>
#include "python_arguments.h"

namespace py = pybind11;
using namespace pybind11::literals;

using esig::deg_t;
using esig::dimn_t;
using esig::param_t;
using esig::key_type;
using esig::interval;
using esig::real_interval;
using esig::dyadic_interval;

using esig::algebra::lie;
using esig::algebra::free_tensor;
using esig::algebra::context;
using esig::algebra::vector_type;
using esig::algebra::coefficient_type;

using esig::paths::path;
using esig::paths::path_interface;
using esig::paths::piecewise_lie_path;

static const char* PLP_DOC = R"epdoc(Piecewise Lie Path
)epdoc";


void esig::paths::init_piecewise_lie_path(pybind11::module_ &m) {
    py::class_<piecewise_lie_path, path_interface> klass(m, "PiecewiseLiePath", PLP_DOC);

    esig::paths::register_pypath_constructor(py::type::of<piecewise_lie_path>(),
        construct_piecewise_lie_path);
}
esig::paths::path esig::paths::construct_piecewise_lie_path(
    const pybind11::args &args, const pybind11::kwargs &kwargs)
{
    std::vector<std::pair<real_interval, lie>> data;

    additional_args helper;

    auto check_lie = [&helper] (const lie& arg) {
        if (helper.width == 0) {
            helper.width = arg.width();
            helper.depth = arg.depth();
            helper.ctype = arg.coeff_type();
        } else {
            if (helper.width != arg.width()) {
                throw py::value_error("lie elements must have the same width");
            }
            if (helper.ctype != arg.coeff_type()) {
                throw py::value_error("lie elements must have a common coefficient type");
            }
            if (helper.depth != arg.depth()) {
                throw py::value_error("lie elements must have a common depth");
            }
        }
    };
    for (const auto& arg : args) {
        if (py::isinstance<py::tuple>(arg)) {
            // Assume (interval, lie)
            try {
                data.push_back(arg.cast<std::pair<real_interval, lie>>());
                check_lie(data.back().second);
            } catch (py::cast_error& err) {
                throw py::value_error("tuple arguments are assumed to be pairs (interval, lie)");
            }
        } else if (py::isinstance<py::iterable>(arg)) {
            auto iterable = arg.cast<py::iterable>();
            data.reserve(data.size() + len_hint(iterable));
            for (auto val : iterable) {
                try {
                    data.push_back(val.cast<std::pair<real_interval, lie>>());
                    check_lie(data.back().second);
                } catch (py::cast_error& err) {
                    throw py::value_error("items should be pairs (interval, lie)");
                }
            }
        } else {
            throw py::type_error("provide either a pair (interval, lie) or an iterable of such pairs");
        }
    }

    auto md = parse_kwargs_to_metadata(kwargs, helper);

    return path(piecewise_lie_path(data, std::move(md)));
}
