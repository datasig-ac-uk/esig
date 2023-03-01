//
// Created by user on 10/02/23.
//

#include "py_context.h"

#ifndef ESIG_NO_NUMPY
#include <pybind11/numpy.h>
#endif
#include <pybind11/stl.h>



#include "scalar_meta.h"
#include "py_lie_key_iterator.h"
#include "py_tensor_key_iterator.h"


using namespace esig;
using namespace esig::algebra;
using namespace esig::python;

namespace py = pybind11;
using namespace py::literals;



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
        request.data_stream.push_back(scalars::ScalarPointer(array.data(i, 0)));
    }
    request.vect_type = vector_type::dense;

    auto sig = ctx->signature(request);

    return sig;
//    return ctx->signature(request);
}
#endif


static py_context py_get_context(deg_t width, deg_t depth, const py::object& ctype, const py::kwargs& kwargs)
{
    //TODO: Make this accept extra arguments.
    return get_context(width, depth, esig::python::to_stype_ptr(ctype), {});
}


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
        "to_logsignature", [](const py_context &ctx, const free_tensor &sig) {
            return ctx->tensor_to_lie(sig.log());
        },
        "signature"_a);


    klass.def("iterate_lie_keys", [](const py_context& ctx) {
        return esig::python::py_lie_key_iterator(&*ctx);
    });
    klass.def("iterate_tensor_keys", [](const py_context& ctx) {
        return esig::python::py_tensor_key_iterator(ctx->width(), ctx->depth());
    });


    m.def("get_context", py_get_context, "width"_a, "depth"_a, "coeffs"_a=py::none());



}
