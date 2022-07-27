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

#include <string>

namespace esig {
namespace algebra {

ESIG_ALGEBRA_EXPORT
pybind11::dtype dtype_from(coefficient_type ctype);

class ESIG_ALGEBRA_EXPORT py_tensor_key
{
    key_type m_key;
    const context* m_ctx;

public:
    explicit py_tensor_key(const context* ctx, key_type key = 0);

    explicit operator key_type() const noexcept;
    const context* get_context() const noexcept;

    std::string to_string() const;
    py_tensor_key lparent() const;
    py_tensor_key rparent() const;

    deg_t degree() const;

    bool equals(const py_tensor_key &other) const noexcept;
    bool less(const py_tensor_key &other) const noexcept;
};

class ESIG_ALGEBRA_EXPORT py_lie_key
{
    key_type m_key;
    const context* m_ctx;

public:

    explicit py_lie_key(const context* ctx, key_type key=1);

    explicit operator key_type() const noexcept;
    const context* get_context() const noexcept;

    std::string to_string() const;
    py_lie_key lparent() const;
    py_lie_key rparent() const;

    deg_t degree() const;

    bool equals(const py_lie_key& other) const noexcept;
    bool less(const py_lie_key& other) const noexcept;

    lie to_lie(const coefficient& c) const;

};

struct py_vector_construction_helper
{
    /// Buffer used if conversion is needed
    allocating_data_buffer buffer;
    /// Context if provided by user
    std::shared_ptr<context> ctx = nullptr;
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


//namespace dtl {
//
//
//template <>
//class coefficient_implementation<pybind11::object> : public coefficient_interface
//{
//    pybind11::object m_data;
//
//    friend struct coefficient_value_helper;
//    using coeff_t = typename pybind11::object;
//    using coeff_impl = coefficient_implementation<pybind11::object>;
//
//    template <typename S>
//    friend class coefficient_implementation;
//
//public:
//
//    explicit coefficient_implementation(pybind11::object arg);
//
//    coefficient_type ctype() const noexcept override;
//    bool is_const() const noexcept override;
//    bool is_val() const noexcept override;
//
//    scalar_t as_scalar() const override;
//    void assign(coefficient val) override;
//    coefficient add(const coefficient_interface &other) const override;
//    coefficient sub(const coefficient_interface &other) const override;
//    coefficient mul(const coefficient_interface &other) const override;
//    coefficient div(const coefficient_interface &other) const override;
//    coefficient add(const scalar_t &other) const override;
//    coefficient sub(const scalar_t &other) const override;
//    coefficient mul(const scalar_t &other) const override;
//    coefficient div(const scalar_t &other) const override;
//};
//
//template <>
//struct coefficient_type_trait<pybind11::object>
//{
//    using value_type = pybind11::object;
//    using reference = value_type;
//    using const_reference = value_type;
//
//    using value_wrapper = coefficient_implementation<value_type>;
//    using reference_wrapper = value_wrapper;
//    using const_reference_wrapper = reference_wrapper;
//
//    static coefficient make(reference arg)
//    {
//        return coefficient(std::shared_ptr<coefficient_interface>(new
//                          value_wrapper(arg)));
//    }
//
//};
//
//
//template <>
//struct coefficient_type_trait<pybind11::object&>
//{
//    using value_type = pybind11::object;
//    using reference = value_type;
//    using const_reference = value_type;
//
//    using value_wrapper = coefficient_implementation<value_type>;
//    using reference_wrapper = value_wrapper;
//    using const_reference_wrapper = value_wrapper;
//
//    static coefficient make(reference arg)
//    {
//        return coefficient(std::shared_ptr<coefficient_interface>(new
//                           value_wrapper(arg)));
//    }
//};
//
//template <>
//struct coefficient_type_trait<const pybind11::object&>
//{
//    using value_type = pybind11::object;
//    using reference = value_type;
//    using const_reference = value_type;
//
//    using value_wrapper = coefficient_implementation<value_type>;
//    using reference_wrapper = value_wrapper;
//    using const_reference_wrapper = value_wrapper;
//
//    static coefficient make(reference arg)
//    {
//        return coefficient(std::shared_ptr<coefficient_interface>(new
//                           value_wrapper(arg)));
//    }
//};
//
//
//
//
//} // namespace dtl
//
//
//namespace dtl {
//
//} // namespace dtl

} // namespace algebra
} // namespace esig

#endif//ESIG_ALGEBRA_PYTHON_INTERFACE_H_
