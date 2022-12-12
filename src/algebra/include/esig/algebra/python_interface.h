//
// Created by user on 26/05/22.
//

#ifndef ESIG_ALGEBRA_PYTHON_INTERFACE_H_
#define ESIG_ALGEBRA_PYTHON_INTERFACE_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/context.h>

#include <boost/container/small_vector.hpp>

#include <string>

namespace esig {
namespace algebra {










ESIG_ALGEBRA_EXPORT
py_vector_construction_helper kwargs_to_construction_data(const pybind11::kwargs& kwargs);







ESIG_ALGEBRA_EXPORT
py_vector_construction_helper
get_construction_data(const pybind11::object& arg,
                      const pybind11::kwargs& kwargs);

namespace dtl {
template<typename... ExtrasFns>
struct exec_for_each;


template <typename Fn, typename... Extras>
struct exec_for_each<Fn, Extras...> : exec_for_each<Extras...>
{
    using next = exec_for_each<Extras...>;

    exec_for_each(Fn&& fn, Extras&&... extras)
        : m_fn(std::move<Fn>(fn)), next(std::move(extras)...)
    {}

    template <typename PyClass>
    void eval(PyClass& m) const
    {
        m_fn(m);
        next::eval(m);
    }
private:
    Fn m_fn;
};

template <>
struct exec_for_each<>
{
    template <typename PyClass>
    void eval(PyClass& m) const
    {}
};


}

template <typename Interface, typename... Extras>
void make_py_wrapper(pybind11::module_& m, const char* name, const char* doc_string, Extras... extras)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    using algebra_type = typename Interface::algebra_t;

    py::class_<algebra_type> klass(m, name, doc_string);

    klass.def_property_readonly("width", &algebra_type::width);
    klass.def_property_readonly("depth", &algebra_type::depth);
    klass.def_property_readonly("dtype", &algebra_type::coeff_type);
    klass.def_property_readonly("storage_type", &algebra_type::storage_type);

    klass.def("size", &algebra_type::size);
    klass.def("degree", &algebra_type::degree);

    klass.def("__getitem__", [](const algebra_type& self, key_type key) {
        return static_cast<scalar_t>(self[key]);
    });
    klass.def("__iter__", [](const algebra_type& self) {
        //TODO: Generic iterators
    });

    klass.def("__neg__", &algebra_type::uminus, py::is_operator());

    klass.def("__add__", &algebra_type::add, py::is_operator());
    klass.def("__sub__", &algebra_type::add, py::is_operator());
    klass.def("__mul__", &algebra_type::smul, py::is_operator());
    klass.def("__truediv__", &algebra_type::sdiv, py::is_operator());
    klass.def("__mul__", &algebra_type::mul, py::is_operator());
    klass.def("__rmul__", &algebra_type::smul, py::is_operator());

    klass.def("__mul__", [](const algebra_type& self, scalar_t arg) {
        return self.smul(scalars::scalar(arg));
    }, py::is_operator());
    klass.def("__mul__", [](const algebra_type& self, long long arg) {
        return self.smul(scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__rmul__", [](const algebra_type& self, scalar_t arg) {
         return self.smul(scalars::scalar(arg));
    }, py::is_operator());
    klass.def("__rmul__", [](const algebra_type& self, long long arg) {
      return self.smul(scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__truediv__", [](const algebra_type& self, scalar_t arg) {
             return self.sdiv(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__truediv__", [](const algebra_type& self, scalar_t arg) {
             return self.sdiv(scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());

    klass.def("__iadd__", &algebra_type::add_inplace, py::is_operator());
    klass.def("__isub__", &algebra_type::sub_inplace, py::is_operator());
    klass.def("__imul__", &algebra_type::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &algebra_type::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &algebra_type::mul_inplace, py::is_operator());

    klass.def("__imul__", [](algebra_type& self, scalar_t arg) {
             return self.smul_inplacei(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__imul__", [](algebra_type& self, long long arg) {
             return self.smul_inplace(scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__itruediv__", [](algebra_type& self, scalar_t arg) {
             return self.sdiv_inplace(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__itruediv__", [](algebra_type& self, long long arg) {
             return self.sdiv_inplace(scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());


    klass.def("add_scal_mul", &algebra_type::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &algebra_type::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &algebra_type::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &algebra_type::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &algebra_type::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &algebra_type::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &algebra_type::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &algebra_type::mul_sdiv, "other"_a, "scalar"_a);


    klass.def("__str__", [](const algebra_type& self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
    });

    klass.def("__repr__", [](const algebra_type& self) {
                std::stringstream ss;
                ss << "Lie(width=" << self.width() << ", depth=" << self.depth();
                ss << ", ctype=" << static_cast<int>(self.coeff_type()) << ')';
                return ss.str();
            });

    klass.def("__eq__",
        [](const algebra_type& lhs, const algebra_type& rhs) { return lhs == rhs; },
        py::is_operator());
    klass.def("__neq__",
              [](const algebra_type& lhs, const algebra_type& rhs) { return lhs != rhs; },
        py::is_operator());

    dtl::exec_for_each<Extras...> helper(std::move(extras)...);
    helper.eval(klass);
}



} // namespace algebra
} // namespace esig

#endif//ESIG_ALGEBRA_PYTHON_INTERFACE_H_
