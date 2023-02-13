//
// Created by sam on 02/05/22.
//

#include "esig/paths/python_interface.h"

#include "esig/paths/lie_increment_path.h"
#include "py_function_path.h"
#include "py_lie_increment_path.h"
#include "py_path.h"
#include "py_piecewise_lie_path.h"
#include "py_tick_path.h"

#include <mutex>



namespace py = pybind11;
using namespace pybind11::literals;

struct type_hash
{
    std::size_t operator()(const pybind11::type& arg) const noexcept
    {
        return std::hash<const void*>{}(arg.ptr());
    }
};


static std::unordered_map<py::type, esig::paths::path_constructor, type_hash> constructor_cache;
static std::mutex lock;

void esig::paths::register_pypath_constructor(const pybind11::type &tp, esig::paths::path_constructor&& ctor)
{

    std::lock_guard<std::mutex> access(lock);
    constructor_cache[tp] = std::move(ctor);
}


esig::paths::path esig::paths::py_path_constructor(const py::args& args, const py::kwargs &kwargs)
{
    std::lock_guard<std::mutex> access(lock);

    auto tp = kwargs["type"];

    if (tp.is_none()) {
        return esig::paths::path(constructor_cache[py::type::of<esig::paths::lie_increment_path>()](args, kwargs));
    } else if (py::isinstance<py::type>(tp)) {
        auto real_type = tp.cast<py::type>();
        auto found = constructor_cache.find(real_type);

        if (found != constructor_cache.end()) {
            return found->second(args, kwargs);
        } else {
            auto obj = real_type(*args, **kwargs);
            if (py::isinstance<esig::paths::path_interface>(obj)) {
                return esig::paths::path(esig::paths::py_path_interface_wrapper(obj));
            }
        }
    }
    throw std::invalid_argument("type must be a valid path_interface");
}

using esig::paths::path_interface;



using lsig_pure = esig::algebra::lie (path_interface::*)(const esig::interval&, const esig::algebra::context&) const;
using lsig_extdyadic = esig::algebra::lie (path_interface::*)(const esig::dyadic_interval&,
                                              typename path_interface::compute_depth_t,
                                              const esig::algebra::context&) const;
using lsig_ext = esig::algebra::lie (path_interface::*)(
        const esig::interval&,
        typename path_interface::compute_depth_t,
        const esig::algebra::context&) const;


void esig::paths::init_python_path_interface(pybind11::module_ &m)
{
    py::class_<esig::paths::path_interface, esig::paths::py_path_interface> klass(m, "PathInterface");
    klass.def("log_signature", static_cast<lsig_pure>(&path_interface::log_signature),
              "domain"_a, "context"_a);
    klass.def("log_signature", static_cast<lsig_extdyadic>(&path_interface::log_signature),
              "domain"_a, "resolution"_a, "context"_a);
    klass.def("log_signature", static_cast<lsig_ext>(&path_interface::log_signature),
              "domain"_a, "resolution"_a, "context"_a);
    klass.def("signature", &path_interface::signature, "domain"_a, "resolution"_a, "context"_a);


    // Initialise the various implementations of the interface
    init_lie_increment_path(m);
    init_py_function_path(m);
    init_tick_path(m);
    init_piecewise_lie_path(m);
}


esig::paths::path_interface::compute_depth_t esig::paths::py_path_interface::compute_depth(esig::accuracy_t accuracy) const noexcept
{
    PYBIND11_OVERRIDE(
            esig::paths::path_interface::compute_depth_t,
            esig::paths::path_interface,
            compute_depth,
            accuracy
            );
}
bool esig::paths::py_path_interface::empty(const esig::interval &domain) const
{
    PYBIND11_OVERRIDE(
            bool,
            esig::paths::path_interface,
            empty,
            domain
            );
}
esig::algebra::lie esig::paths::py_path_interface::log_signature(const esig::interval &domain, const esig::algebra::context &ctx) const
{
    PYBIND11_OVERRIDE_PURE(
            esig::algebra::lie,
            esig::paths::path_interface,
            log_signature,
            domain,
            ctx
            );
}
esig::algebra::lie esig::paths::py_path_interface::log_signature(const esig::dyadic_interval &domain, esig::paths::path_interface::compute_depth_t resolution, const esig::algebra::context &ctx) const
{
    PYBIND11_OVERRIDE(esig::algebra::lie, esig::paths::path_interface, log_signature,
                      domain, resolution, ctx);
}
esig::algebra::lie esig::paths::py_path_interface::log_signature(const esig::interval &domain, esig::paths::path_interface::compute_depth_t resolution, const esig::algebra::context &ctx) const
{
    PYBIND11_OVERRIDE(esig::algebra::lie, esig::paths::path_interface, log_signature,
                      domain, resolution, ctx);
}
esig::algebra::free_tensor esig::paths::py_path_interface::signature(const esig::interval &domain, esig::paths::path_interface::compute_depth_t resolution, const esig::algebra::context &ctx) const
{
        PYBIND11_OVERRIDE(esig::algebra::free_tensor, esig::paths::path_interface, signature,
                          domain, resolution, ctx);
}
esig::paths::py_path_interface_wrapper::py_path_interface_wrapper(pybind11::object wrap)
        : path_interface(py::cast<const esig::paths::path_interface &>(wrap).metadata()),
          obj(std::move(wrap))
{
}
esig::algebra::lie esig::paths::py_path_interface_wrapper::log_signature(const esig::interval &domain, const esig::algebra::context &ctx) const
{
    return py::cast<const esig::paths::path_interface &>(obj).log_signature(domain, ctx);
}
