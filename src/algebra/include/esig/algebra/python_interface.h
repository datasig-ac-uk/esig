//
// Created by user on 26/05/22.
//

#ifndef ESIG_ALGEBRA_PYTHON_INTERFACE_H_
#define ESIG_ALGEBRA_PYTHON_INTERFACE_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/context.h>

#include <boost/container/small_vector.hpp>

#include <string>

namespace esig {
namespace algebra {

ESIG_ALGEBRA_EXPORT
pybind11::dtype dtype_from(coefficient_type ctype);

class ESIG_ALGEBRA_EXPORT py_tensor_key
{
    key_type m_key;
    deg_t m_width;
    deg_t m_depth;

public:
    explicit py_tensor_key(key_type key, deg_t width, deg_t depth);

    explicit operator key_type() const noexcept;

    std::string to_string() const;
    py_tensor_key lparent() const;
    py_tensor_key rparent() const;
    bool is_letter() const;

    deg_t width() const;
    deg_t depth() const;

    deg_t degree() const;
    std::vector<let_t> to_letters() const;

    bool equals(const py_tensor_key &other) const noexcept;
    bool less(const py_tensor_key &other) const noexcept;
};


class ESIG_ALGEBRA_EXPORT py_lie_letter
{
    dimn_t m_data = 0;

    constexpr explicit py_lie_letter(dimn_t raw) : m_data(raw)
    {}

public:

    py_lie_letter() = default;

    static constexpr py_lie_letter from_letter(let_t letter) {
        return py_lie_letter(1 + (dimn_t(letter) << 1));
    }

    static constexpr py_lie_letter from_offset(dimn_t offset) {
        return py_lie_letter(offset << 1);
    }

    constexpr bool is_offset() const noexcept
    {
        return (m_data & 1) == 0;
    }

    explicit operator let_t () const noexcept
    {
        return esig::let_t(m_data >> 1);
    }

    explicit constexpr operator dimn_t() const noexcept
    {
        return m_data >> 1;
    }

    inline friend std::ostream& operator<<(std::ostream& os, const py_lie_letter& let)
    {
        return os << let.m_data;
    }

};


class ESIG_ALGEBRA_EXPORT py_lie_key
{
public:
    using container_type = boost::container::small_vector<py_lie_letter, 2>;

private:

    container_type m_data;
    deg_t m_width;


public:

    explicit py_lie_key(deg_t width);
    explicit py_lie_key(deg_t width, let_t letter);
    explicit py_lie_key(deg_t width, const boost::container::small_vector_base<py_lie_letter>& data);
    explicit py_lie_key(deg_t width, let_t left, let_t right);
    explicit py_lie_key(deg_t width, let_t left, const py_lie_key& right);
    explicit py_lie_key(deg_t width, const py_lie_key& left, const py_lie_key& right);

    bool is_letter() const noexcept;
    let_t as_letter() const;
    std::string to_string() const;
    py_lie_key lparent() const;
    py_lie_key rparent() const;

    deg_t degree() const;

    bool equals(const py_lie_key& other) const noexcept;


};

struct py_vector_construction_helper
{
    /// Buffer used if conversion is needed
    allocating_data_buffer buffer;
    /// Context if provided by user
    std::shared_ptr<const context> ctx = nullptr;
    /// Pointers to beginning and end of data
    const char* begin_ptr;
    const char* end_ptr;
    /// Number of elements
    dimn_t count = 0;
    /// Item size
    dimn_t itemsize = 0;
    /// Width and depth
    deg_t width = 0;
    deg_t depth = 0;
    /// Coefficient type
    coefficient_type ctype = coefficient_type::dp_real;
    /// Vector type to be requested
    vector_type vtype = vector_type::dense;
    /// flags for saying if the user explicitly requested ctype and vtype
    bool ctype_requested = false;
    bool vtype_requested = false;
    /// Data type provided
    input_data_type input_vec_type = input_data_type::value_array;
};

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
        return self.smul(coefficient(arg));
    }, py::is_operator());
    klass.def("__mul__", [](const algebra_type& self, long long arg) {
        return self.smul(coefficient(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__rmul__", [](const algebra_type& self, scalar_t arg) {
         return self.smul(coefficient(arg));
    }, py::is_operator());
    klass.def("__rmul__", [](const algebra_type& self, long long arg) {
      return self.smul(coefficient(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__truediv__", [](const algebra_type& self, scalar_t arg) {
             return self.sdiv(coefficient(arg));
         }, py::is_operator());
    klass.def("__truediv__", [](const algebra_type& self, scalar_t arg) {
             return self.sdiv(coefficient(arg, 1LL, self.coeff_type()));
         }, py::is_operator());

    klass.def("__iadd__", &algebra_type::add_inplace, py::is_operator());
    klass.def("__isub__", &algebra_type::sub_inplace, py::is_operator());
    klass.def("__imul__", &algebra_type::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &algebra_type::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &algebra_type::mul_inplace, py::is_operator());

    klass.def("__imul__", [](algebra_type& self, scalar_t arg) {
             return self.smul_inplace(coefficient(arg));
         }, py::is_operator());
    klass.def("__imul__", [](algebra_type& self, long long arg) {
             return self.smul_inplace(coefficient(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__itruediv__", [](algebra_type& self, scalar_t arg) {
             return self.sdiv_inplace(coefficient(arg));
         }, py::is_operator());
    klass.def("__itruediv__", [](algebra_type& self, long long arg) {
             return self.sdiv_inplace(coefficient(arg, 1LL, self.coeff_type()));
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
