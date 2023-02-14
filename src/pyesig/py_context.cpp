//
// Created by user on 10/02/23.
//

#include "py_context.h"

#ifndef ESIG_NO_NUMPY
#include <pybind11/numpy.h>
#endif
#include <pybind11/stl.h>

#include <esig/algebra/context.h>

#include "scalar_meta.h"

using namespace esig;
using namespace esig::algebra;

namespace py = pybind11;
using namespace py::literals;


namespace {

class py_context
{
    std::shared_ptr<const context> p_ctx;

public:

    py_context(std::shared_ptr<const context> ctx) : p_ctx(ctx) {}


    operator const std::shared_ptr<const context>& () const noexcept { return p_ctx; }

    const context& operator*() const noexcept { return *p_ctx; }
    const context* operator->() const noexcept { return p_ctx.get(); }

};


}

#ifndef ESIG_NO_NUMPY
static free_tensor context_compute_signature_numpy_darray(const py_context &ctx, const py::array_t<double, py::array::forcecast> &array) {
    assert(array.ndim() == 2);
    auto shape = array.shape();

    const auto n_increments = shape[0];
    const auto width = shape[1];

    assert(width == ctx->width());

    signature_data request;
    request.data_stream.set_ctype(ctx->ctype());
    request.data_stream.set_elts_per_row(width);
    request.data_stream.reserve_size(n_increments);
    for (dimn_t i = 0; i < n_increments; ++i) {
        request.data_stream.push_back(scalars::scalar_pointer(array.data(i, 0)));
    }
    request.vect_type = vector_type::dense;

    auto sig = ctx->signature(request);

    return sig;
//    return ctx->signature(request);
}
#endif


void esig::python::init_context(py::module_& m) {

    py::class_<py_context> klass(m, "Context");

    klass.def_property_readonly("width", [](const py_context& ctx) { return ctx->width(); });
    klass.def_property_readonly("depth", [](const py_context& ctx) { return ctx->depth(); });
    klass.def_property_readonly("ctype", [](const py_context& ctx) { return to_ctype_type(ctx->ctype()); });
    klass.def("lie_size", [](const py_context& ctx, deg_t degree) { return ctx->lie_size(degree); }, "degree"_a);
    klass.def("tensor_size", [](const py_context& ctx, deg_t degree) { return ctx->tensor_size(degree); }, "degree"_a);
    klass.def("cbh", [](const py_context& ctx, std::vector<lie> lies, algebra::vector_type vtype) {
        return ctx->cbh(lies, vtype);
    }, "lies"_a, "vec_type"_a);

#ifndef ESIG_NO_NUMPY
    klass.def("compute_signature", context_compute_signature_numpy_darray, "data"_a);
#endif
    klass.def(
        "to_logsig", [](const py_context &ctx, const free_tensor &sig) {
            return ctx->tensor_to_lie(sig.log());
        },
        "signature"_a);

    m.def("get_context",
        [](deg_t width, deg_t depth, const py::object& ctype_c, std::vector<std::string> other) {
            return py_context(get_context(width, depth, to_stype_ptr(ctype_c), other));
        }, "width"_a, "depth"_a, "coeffs"_a, "other"_a);


}
