//
// Created by sam on 04/01/23.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_LITE_CONTEXT_H_
#define ESIG_SRC_ALGEBRA_SRC_LITE_CONTEXT_H_

#include "esig/algebra/context.h"

#include <functional>
#include <memory>
#include <vector>
#include <ostream>

#include <boost/move/utility.hpp>
#include <boost/move/utility_core.hpp>
#include <libalgebra_lite/dense_vector.h>
#include <libalgebra_lite/free_tensor.h>
#include <libalgebra_lite/hall_set.h>
#include <libalgebra_lite/lie.h>
#include <libalgebra_lite/maps.h>
#include <libalgebra_lite/sparse_vector.h>
#include <libalgebra_lite/tensor_basis.h>
#include <libalgebra_lite/registry.h>


#include "esig/algebra/algebra_traits.h"

namespace esig {
namespace algebra {


namespace dtl {

template <typename VT>
using storage_type = lal::dtl::standard_storage<VT>;

template <vector_type>
struct vector_type_selector;

template <>
struct vector_type_selector<vector_type::dense>
{
    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::dense_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::dense_vector, storage_type>;
};

template <>
struct vector_type_selector<vector_type::sparse>
{
    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::sparse_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::sparse_vector, storage_type>;
};





}



template <typename Coefficients>
class lite_context : public context {

    deg_t m_width;
    deg_t m_depth;
    using coefficients_type = Coefficients;

    std::shared_ptr<const lal::hall_basis> p_lie_basis;
    std::shared_ptr<const lal::tensor_basis> p_tensor_basis;

    std::shared_ptr<const lal::free_tensor_multiplication> p_ftmul;
    std::shared_ptr<const lal::lie_multiplication> p_liemul;

    template <vector_type VType>
    using free_tensor_t = typename dtl::vector_type_selector<VType>::template free_tensor<Coefficients>;

    template <vector_type VType>
    using lie_t = typename dtl::vector_type_selector<VType>::template lie<Coefficients>;

    using ctype_traits = lal::coefficient_trait<Coefficients>;

    using scalar_type = typename ctype_traits::scalar_type;
    using rational_type = typename ctype_traits::rational_type;

    lal::maps m_maps;


    template <typename OutType, typename InType>
    OutType convertImpl(const InType& arg,
                        const std::shared_ptr<const typename OutType::basis_type>& basis,
                        const std::shared_ptr<const typename OutType::multiplication_type>& mul) const;

    template <typename OutType>
    OutType constructImpl(const vector_construction_data& data,
                          const std::shared_ptr<const typename OutType::basis_type>& basis,
                          const std::shared_ptr<const typename OutType::multiplication_type>& mul) const;

    template <vector_type VType>
    free_tensor_t<VType> lie_to_tensor_impl(const lie& arg) const;

    template <vector_type VType>
    lie_t<VType> tensor_to_lie_impl(const free_tensor& arg) const;

    template <vector_type VType>
    lie cbhImpl(const std::vector<lie>& lies) const;

    template <vector_type VType>
    free_tensor_t<VType> compute_signature(const signature_data& data) const;

    template <vector_type VType>
    free_tensor_t<VType> Ad_x_n(deg_t d, const free_tensor_t<VType>& x, const free_tensor_t<VType>& y) const;

    template <vector_type VType>
    free_tensor_t <VType> derive_series_compute(const free_tensor_t<VType>& increment, const free_tensor_t<VType>& perturbation) const;

    template <vector_type VType>
    free_tensor_t<VType> sig_derivative_single(const free_tensor_t<VType>& signature, const free_tensor_t<VType>& tincr, const free_tensor_t<VType>& perturbation) const;

    template <vector_type VType>
    free_tensor_t<VType> sig_derivative_impl(const std::vector<derivative_compute_info>& info) const;

public:

    lite_context(deg_t width, deg_t depth);

    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    std::shared_ptr<const context> get_alike(deg_t new_depth) const override;
    std::shared_ptr<const context> get_alike(const scalars::scalar_type *new_coeff) const override;
    std::shared_ptr<const context> get_alike(deg_t new_depth, const scalars::scalar_type *new_coeff) const override;
    std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, const scalars::scalar_type *new_coeff) const override;
    dimn_t lie_size(deg_t d) const noexcept override;
    dimn_t tensor_size(deg_t d) const noexcept override;
    bool check_compatible(const context &other) const noexcept override;
    free_tensor convert(const free_tensor &arg, vector_type new_vec_type) const override;
    lie convert(const lie &arg, vector_type new_vec_type) const override;
    basis get_tensor_basis() const override;
    basis get_lie_basis() const override;
    free_tensor construct_tensor(const vector_construction_data &data) const override;
    lie construct_lie(const vector_construction_data &data) const override;
    free_tensor lie_to_tensor(const lie &arg) const override;
    lie tensor_to_lie(const free_tensor &arg) const override;
    lie cbh(const std::vector<lie> &lies, vector_type vtype) const override;
    free_tensor signature(const signature_data &data) const override;
    lie log_signature(const signature_data &data) const override;
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const override;
};


class lite_context_maker : public context_maker {
public:
    bool can_get(deg_t deg, deg_t deg_1, const scalars::scalar_type *type) const noexcept override;
    int get_priority(const std::vector<std::string> &preferences) const noexcept override;
    std::shared_ptr<const context> get_context(deg_t deg, deg_t deg_1, const scalars::scalar_type *type) const override;
};



template<typename Coefficients>
lite_context<Coefficients>::lite_context(deg_t width, deg_t depth)
    : context(scalars::dtl::scalar_type_holder<scalar_type>::get_type()),
      m_width(width), m_depth(depth),
      p_lie_basis(lal::basis_registry<lal::hall_basis>::get(lal::deg_t(width), lal::deg_t(depth))),
      p_tensor_basis(lal::basis_registry<lal::tensor_basis>::get(lal::deg_t(width), lal::deg_t(depth))),
      m_maps(p_tensor_basis, p_lie_basis)
{
}

template<typename Coefficients>
deg_t lite_context<Coefficients>::width() const noexcept {
    return m_width;
}
template<typename Coefficients>
deg_t lite_context<Coefficients>::depth() const noexcept {
    return m_depth;
}
template<typename Coefficients>
std::shared_ptr<const context> lite_context<Coefficients>::get_alike(deg_t new_depth) const {
    lite_context_maker maker;
    return maker.get_context(m_width, new_depth, context::ctype());
}
template<typename Coefficients>
std::shared_ptr<const context> lite_context<Coefficients>::get_alike(const scalars::scalar_type *new_coeff) const {
    lite_context_maker maker;
    return maker.get_context(m_width, m_depth, new_coeff);
}
template<typename Coefficients>
std::shared_ptr<const context> lite_context<Coefficients>::get_alike(deg_t new_depth, const scalars::scalar_type *new_coeff) const {
    lite_context_maker maker;
    return maker.get_context(m_width, new_depth, new_coeff);
}
template<typename Coefficients>
std::shared_ptr<const context> lite_context<Coefficients>::get_alike(deg_t new_width, deg_t new_depth, const scalars::scalar_type *new_coeff) const {
    lite_context_maker maker;
    return maker.get_context(new_width, new_depth, new_coeff);
}
template<typename Coefficients>
dimn_t lite_context<Coefficients>::lie_size(deg_t d) const noexcept {
    return p_lie_basis->size(lal::deg_t(d));
}
template<typename Coefficients>
dimn_t lite_context<Coefficients>::tensor_size(deg_t d) const noexcept {
    return p_tensor_basis->size(lal::deg_t(d));
}
template<typename Coefficients>
bool lite_context<Coefficients>::check_compatible(const context &other) const noexcept {
    return false;
}

template<typename Coefficients>
template<typename OutType, typename InType>
OutType lite_context<Coefficients>::convertImpl(const InType &arg,
                                                const std::shared_ptr<const typename OutType::basis_type> &basis,
                                                const std::shared_ptr<const typename OutType::multiplication_type> &mul) const {
    OutType result(basis, mul);
    return result;
}
template<typename Coefficients>
free_tensor lite_context<Coefficients>::convert(const free_tensor &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(convertImpl<free_tensor_t<(VTYPE)>>(arg, p_tensor_basis, p_ftmul), this)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}
template<typename Coefficients>
lie lite_context<Coefficients>::convert(const lie &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) lie(convertImpl<lie_t<(VTYPE)>>(arg, p_lie_basis, p_liemul), this)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}
template<typename Coefficients>
basis lite_context<Coefficients>::get_tensor_basis() const {
    return basis(p_tensor_basis);
}
template<typename Coefficients>
basis lite_context<Coefficients>::get_lie_basis() const {
    return basis(p_lie_basis);
}

template<typename Coefficients>
template<typename OutType>
OutType lite_context<Coefficients>::constructImpl(const vector_construction_data &data,
                                                  const std::shared_ptr<const typename OutType::basis_type>& basis,
                                                  const std::shared_ptr<const typename OutType::multiplication_type>& mul) const {
    OutType result(basis);

    if (data.data.is_null()) {
        return result;
    }

    const scalar_type* data_ptr;


    const auto size = data.data.size();
    std::vector<scalar_type> tmp;
    if (data.data.type() != ctype()) {
        tmp.resize(data.data.size());
        ctype()->convert_copy(tmp.data(), data.data, size);
        data_ptr = tmp.data();
    } else {
        data_ptr = data.data.raw_cast<const scalar_type>();
    }

    if (data.data.has_keys()) {
        // Sparse data
        const auto* keys = data.data.keys();

        for (dimn_t i=0; i<size; ++i) {
            result[basis->index_to_key(i)] = data_ptr[i];
        }

    } else {
        // Dense data

        for (dimn_t i=0; i<size; ++i) {
            // Replace this with a more efficient method once it's implemented at the lower level
            result[basis->index_to_key(i)] = data_ptr[i];
        }
    }

    return result;
}

template<typename Coefficients>
free_tensor lite_context<Coefficients>::construct_tensor(const vector_construction_data &data) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(constructImpl<free_tensor_t<(VTYPE)>>(data, p_tensor_basis, p_ftmul), this)
    ESIG_MAKE_VTYPE_SWITCH(data.vect_type)
#undef ESIG_SWITCH_FN
}
template<typename Coefficients>
lie lite_context<Coefficients>::construct_lie(const vector_construction_data &data) const {
#define ESIG_SWITCH_FN(VTYPE) lie(constructImpl<lie_t<(VTYPE)>>(data, p_lie_basis, p_liemul), this)
    ESIG_MAKE_VTYPE_SWITCH(data.vect_type)
#undef ESIG_SWITCH_FN
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::lie_to_tensor_impl(const lie &arg) const {
    const auto* interface = algebra_access<lie_interface>::get(arg);

    if (interface->id() == reinterpret_cast<std::uintptr_t>(this)) {
        return m_maps.template lie_to_tensor(algebra_access<lie_interface>::template get<lie_t<VType>>(arg));
    }

    free_tensor tmp_tensor(free_tensor_t<VType>(p_tensor_basis, p_ftmul), this);
    context::lie_to_tensor_fallback(tmp_tensor, arg);
    return algebra_access<free_tensor_interface>::template get<free_tensor_t<VType>>(tmp_tensor);
}


template<typename Coefficients>
free_tensor lite_context<Coefficients>::lie_to_tensor(const lie &arg) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(lie_to_tensor_impl<(VTYPE)>(arg), this);
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template lie_t<VType>
lite_context<Coefficients>::tensor_to_lie_impl(const free_tensor &arg) const {
    const auto* interface = algebra_access<free_tensor_interface>::get(arg);

    if (interface->id() == reinterpret_cast<std::uintptr_t>(this)) {
        return m_maps.template tensor_to_lie(algebra_access<free_tensor_interface>::template get<free_tensor_t<VType>>(arg));
    }

    lie tmp(lie_t<VType>(p_lie_basis, p_liemul), this);
    context::tensor_to_lie_fallback(tmp, arg);
    return algebra_access<lie_interface>::template get<lie_t<VType>>(tmp);
}

template<typename Coefficients>
lie lite_context<Coefficients>::tensor_to_lie(const free_tensor &arg) const {
#define ESIG_SWITCH_FN(VTYPE) lie(tensor_to_lie_impl<VTYPE>(arg), this)
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
}

template<typename Coefficients>
template<vector_type VType>
lie lite_context<Coefficients>::cbhImpl(const std::vector<lie> &lies) const {
    free_tensor tmp(free_tensor_t<VType>(p_tensor_basis, p_ftmul), this);
    context::cbh_fallback(tmp, lies);
    return tensor_to_lie(tmp.log());
}

template<typename Coefficients>
lie lite_context<Coefficients>::cbh(const std::vector<lie> &lies, vector_type vtype) const {
#define ESIG_SWITCH_FN(VTYPE) cbhImpl<(VTYPE)>(lies)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::compute_signature(const signature_data &data) const {

    free_tensor_t<VType> result(p_tensor_basis, p_ftmul);
    result[typename lal::tensor_basis::key_type()] = scalar_type(1);
    const auto nrows = data.data_stream.row_count();

    for (dimn_t i=0; i<nrows; ++i) {
        auto row = data.data_stream[i];
        const auto* keys = data.key_stream.empty() ? nullptr : data.key_stream[i];
        vector_construction_data row_cdata { scalars::key_scalar_array(row, keys), VType };

        auto lie_row = constructImpl<lie_t<VType>>(row_cdata, p_lie_basis, p_liemul);

#if 0
        result.fmexp_inplace(m_maps.lie_to_tensor(lie_row));
#endif
    }

    return result;
}

template<typename Coefficients>
free_tensor lite_context<Coefficients>::signature(const signature_data &data) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(compute_signature<VTYPE>(data), this)
    ESIG_MAKE_VTYPE_SWITCH(data.vect_type)
#undef ESIG_SWITCH_FN
}
template<typename Coefficients>
lie lite_context<Coefficients>::log_signature(const signature_data &data) const {
    return tensor_to_lie(signature(data).log());
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::Ad_x_n(deg_t d, const free_tensor_t<VType>& x, const free_tensor_t<VType>& y) const
{
    auto tmp = x * y - y * x;
    while (--d) {
        tmp = x * tmp - tmp * x;
    }
    return tmp;
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::derive_series_compute(const free_tensor_t<VType> &increment, const free_tensor_t<VType>& perturbation) const {
    free_tensor_t<VType> result(perturbation);

    typename Coefficients::scalar_type factor(1);
    for (deg_t d=1; d <= m_depth; ++d) {
        factor *= typename Coefficients::scalar_type(d+1);
        if (d % 2 == 0) {
            result.add_scal_div(Ad_x_n<VType>(d, increment, perturbation), factor);
        } else {
            result.sub_scal_div(Ad_x_n<VType>(d, increment, perturbation), factor);
        }
    }
    return result;
};

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::sig_derivative_single(const free_tensor_t<VType>& signature, const free_tensor_t<VType>& tincr, const free_tensor_t<VType>& perturbation) const {
    return signature * derive_series_compute<VType>(tincr, perturbation);
}

template<typename Coefficients>
template<vector_type VType>
typename lite_context<Coefficients>::template free_tensor_t<VType>
lite_context<Coefficients>::sig_derivative_impl(const std::vector<derivative_compute_info> &info) const {
    using tensor_type = free_tensor_t<VType>;
    using lie_type = lie_t<VType>;

    if (info.empty()) {
        return tensor_type(p_tensor_basis, p_ftmul);
    }

    using access = algebra_access<lie_interface>;

    tensor_type result(p_tensor_basis, p_ftmul);

    for (const auto& data : info) {
        auto tincr = lie_to_tensor_impl<VType>(data.logsig_of_interval);
        auto tperturb = lie_to_tensor_impl<VType>(data.perturbation);
        auto signature = exp(tincr);

        result *= signature;
        result += sig_derivative_single<VType>(signature, tincr, tperturb);
    }

    return result;
}

template<typename Coefficients>
free_tensor lite_context<Coefficients>::sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(sig_derivative_impl<(VTYPE)>(info), this)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}

}// namespace algebra

/*
 * Now we need to define all the traits and interface elements so that esig can successfully communicate
 * with the libalgebra_lite types. We need:
 *   - scalar types mappings for the sparse reference types
 *   - algebra info for the various algebra types
 *   - basis wrappers
 *   - Iterator traits
 */

namespace scalars {
namespace dtl {

template <typename MapType, typename KeyType>
class scalar_type_trait<lal::dtl::sparse_mutable_reference<MapType, KeyType>>
{
public:

    using value_type = typename MapType::mapped_type;
    using rational_type = value_type;
    using reference = lal::dtl::sparse_mutable_reference<MapType, KeyType>;

    static const scalar_type* get_type() noexcept { return scalar_type_holder<value_type>::get_type(); }

    static scalar make(reference val) {
        return scalar (new scalar_implementation<reference>(std::move(val)));
    }
};


template <typename MapType, typename KeyType>
class scalar_implementation<lal::dtl::sparse_mutable_reference<MapType, KeyType>>
    : public scalar_interface
{
    using data_type = lal::dtl::sparse_mutable_reference<MapType, KeyType>;
    using trait = scalar_type_trait<data_type>;

    data_type m_data;

public:
    using value_type = typename MapType::mapped_type;
    using rational_type = typename trait::rational_type;

    explicit scalar_implementation(data_type&& d) : m_data(std::move(d))
    {}

    const scalar_type* type() const noexcept override { return trait::get_type(); }

    bool is_const() const noexcept override { return false; }
    bool is_value() const noexcept override { return false; }
    bool is_zero() const noexcept override {
        return static_cast<const value_type&>(m_data) == value_type(0);
    }
    scalar_t as_scalar() const override {
        return scalar_t(static_cast<const value_type&>(m_data));
    }
    void assign(scalar_pointer other) override {
        value_type tmp = static_cast<const value_type&>(m_data);
        this->type()->convert_copy(&tmp, other, 1);
        m_data = tmp;
    }
    void assign(const scalar &other) override {
        this->assign(other.to_const_pointer());
    }
    scalar_pointer to_pointer() override {
        throw std::runtime_error("cannot get non-const pointer to proxy reference type");
    }
    scalar_pointer to_pointer() const noexcept override {
        return scalar_pointer(&static_cast<const value_type&>(m_data), this->type());
    }
    scalar uminus() const override {
        return scalar(-static_cast<const value_type&>(m_data), this->type());
    }

private:

    template <typename Fn>
    void inplace_function(const scalar& other, Fn f)
    {
        value_type tmp(0);
        this->type()->convert_copy(&tmp, other.to_const_pointer(), 1);
        m_data = f(static_cast<const value_type&>(m_data), tmp);
    }

public:
    void add_inplace(const scalar &other) override {
        inplace_function(other, std::plus<value_type>());
    }
    void sub_inplace(const scalar &other) override {
        inplace_function(other, std::minus<value_type>());
    }
    void mul_inplace(const scalar &other) override {
        inplace_function(other, std::multiplies<value_type>());
    }
    void div_inplace(const scalar &other) override {
        rational_type tmp(1);
        this->type()->rational_type()->convert_copy(&tmp, other.to_const_pointer(), 1);
        m_data /= tmp;
    }
    bool equals(const scalar &other) const noexcept override {
        value_type tmp(0);
        this->type()->convert_copy(&tmp, other.to_const_pointer(), 1);
        return static_cast<const value_type&>(m_data) == tmp;
    }
    std::ostream &print(std::ostream &os) const override {
        return os << static_cast<const value_type&>(m_data);
    }
};


} // namespace dtl
} // namespace scalars


namespace algebra {


template <typename Coeffs>
struct algebra_info<lal::free_tensor<Coeffs, lal::sparse_vector, lal::dtl::standard_storage>>
{
    using algebra_type = lal::free_tensor<Coeffs, lal::sparse_vector, lal::dtl::standard_storage>;
    using basis_type = lal::tensor_basis;
    using this_key_type = typename basis_type::key_type;

    using scalar_type = typename Coeffs::scalar_type;
    using rational_type = typename Coeffs::rational_type;
    using reference = decltype(std::declval<algebra_type>()[std::declval<this_key_type>()]);
    using const_reference = const scalar_type&;

    static const scalars::scalar_type* ctype() noexcept
    { return scalars::dtl::scalar_type_holder<scalar_type>::get_type(); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::sparse; }
    static deg_t width(const algebra_type* instance) noexcept
    { return instance->basis().width(); }
    static deg_t max_depth(const algebra_type* instance) noexcept
    { return instance->basis().depth(); }

    static const basis_type &basis(const algebra_type &instance) noexcept { return instance.basis(); }
    static this_key_type convert_key(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().index_to_key(key); }

    static deg_t degree(const algebra_type& instance) noexcept
//    { return instance.degree(); }
    { return 0; }
    static deg_t degree(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const algebra_type* instance, this_key_type key)
    { return instance->basis().degree(key); }

    static key_type first_key(const algebra_type* instance) noexcept
    { return 0; }
    static key_type last_key(const algebra_type* instance) noexcept
    { return instance->basis().size(); }


    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container& key_product(const algebra_type* instance, key_type k1, key_type k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container &key_product(const algebra_type *instance, const this_key_type &k1, const this_key_type &k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type(instance.get_basis());
    }

};


template <typename Coeffs>
struct algebra_info<lal::free_tensor<Coeffs, lal::dense_vector, lal::dtl::standard_storage>>
{
    using algebra_type = lal::free_tensor<Coeffs, lal::dense_vector, lal::dtl::standard_storage>;
    using basis_type = lal::tensor_basis;
    using this_key_type = typename basis_type::key_type;

    using scalar_type = typename Coeffs::scalar_type;
    using rational_type = typename Coeffs::rational_type;
    using reference = decltype(std::declval<algebra_type>()[std::declval<this_key_type>()]);
    using const_reference = const scalar_type&;

    static const scalars::scalar_type* ctype() noexcept
    { return scalars::dtl::scalar_type_holder<scalar_type>::get_type(); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::dense; }
    static deg_t width(const algebra_type* instance) noexcept
    { return instance->basis().width(); }
    static deg_t max_depth(const algebra_type* instance) noexcept
    { return instance->basis().depth(); }

    static const basis_type &basis(const algebra_type &instance) noexcept { return instance.basis(); }
    static this_key_type convert_key(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().index_to_key(key); }

    static deg_t degree(const algebra_type& instance) noexcept
    { return /*instance.degree()*/ 0; }
    static deg_t degree(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const algebra_type* instance, this_key_type key)
    { return instance->basis().degree(key); }

    static key_type first_key(const algebra_type* instance) noexcept
    { return 0; }
    static key_type last_key(const algebra_type* instance) noexcept
    { return instance->basis().size(); }


    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container& key_product(const algebra_type* instance, key_type k1, key_type k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container &key_product(const algebra_type *instance, const this_key_type &k1, const this_key_type &k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type(instance.get_basis());
//        return algebra_type();
    }

};

template <typename Coeffs>
struct algebra_info<lal::lie<Coeffs, lal::sparse_vector, lal::dtl::standard_storage>>
{
    using algebra_type = lal::lie<Coeffs, lal::sparse_vector, lal::dtl::standard_storage>;
    using basis_type = lal::hall_basis;
    using this_key_type = typename basis_type::key_type;

    using scalar_type = typename Coeffs::scalar_type;
    using rational_type = typename Coeffs::rational_type;
    using reference = decltype(std::declval<algebra_type>()[std::declval<this_key_type>()]);
    using const_reference = const scalar_type&;

    static const scalars::scalar_type* ctype() noexcept
    { return scalars::dtl::scalar_type_holder<scalar_type>::get_type(); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::sparse; }
    static deg_t width(const algebra_type* instance) noexcept
    { return instance->basis().width(); }
    static deg_t max_depth(const algebra_type* instance) noexcept
    { return instance->basis().depth(); }

    static const basis_type &basis(const algebra_type &instance) noexcept { return instance.basis(); }

    static this_key_type convert_key(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().index_to_key(key-1); }

    static deg_t degree(const algebra_type& instance) noexcept
    { return /*instance.degree()*/ 0; }
    static deg_t degree(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const algebra_type* instance, this_key_type key)
    { return instance->basis().degree(key); }

    static key_type first_key(const algebra_type* instance) noexcept
    { return 0; }
    static key_type last_key(const algebra_type* instance) noexcept
    { return instance->basis().size(); }


    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container& key_product(const algebra_type* instance, key_type k1, key_type k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container& key_product(const algebra_type* instance, const this_key_type& k1, const this_key_type& k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }

    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type(instance.get_basis());
    }

};


template <typename Coeffs>
struct algebra_info<lal::lie<Coeffs, lal::dense_vector, lal::dtl::standard_storage>>
{
    using algebra_type = lal::lie<Coeffs, lal::dense_vector, lal::dtl::standard_storage>;
    using basis_type = lal::hall_basis;
    using this_key_type = typename basis_type::key_type;

    using scalar_type = typename Coeffs::scalar_type;
    using rational_type = typename Coeffs::rational_type;
    using reference = decltype(std::declval<algebra_type>()[std::declval<this_key_type>()]);
    using const_reference = const scalar_type&;

    static const scalars::scalar_type* ctype() noexcept
    { return scalars::dtl::scalar_type_holder<scalar_type>::get_type(); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::dense; }
    static deg_t width(const algebra_type* instance) noexcept
    { return instance->basis().width(); }
    static deg_t max_depth(const algebra_type* instance) noexcept
    { return instance->basis().depth(); }

    static const basis_type& basis(const algebra_type& instance) noexcept
    { return instance.basis(); }

    static this_key_type convert_key(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().index_to_key(key-1); }

    static deg_t degree(const algebra_type& instance) noexcept
    { return /*instance.degree()*/ 0; }
    static deg_t degree(const algebra_type* instance, esig::key_type key) noexcept
    { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const algebra_type* instance, this_key_type key)
    { return instance->basis().degree(key); }

    static key_type first_key(const algebra_type* instance) noexcept
    { return 0; }
    static key_type last_key(const algebra_type* instance) noexcept
    { return instance->basis().size(); }


    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container& key_product(const algebra_type* instance, key_type k1, key_type k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container &key_product(const algebra_type *instance, const this_key_type &k1, const this_key_type &k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }

    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type(instance.get_basis());
    }

};




namespace dtl {

template <>
typename lal::tensor_basis::key_type
basis_info<lal::tensor_basis>::convert_key(const lal::tensor_basis &basis, esig::key_type key)
{
    return basis.index_to_key(key);
}
template <>
esig::key_type
basis_info<lal::tensor_basis>::convert_key(const lal::tensor_basis &basis, const this_key_type& key)
{
    return basis.key_to_index(key);
}
template <>
esig::key_type
basis_info<lal::tensor_basis>::first_key(const lal::tensor_basis &basis) { return 0; }
template <>
esig::key_type
basis_info<lal::tensor_basis>::last_key(const lal::tensor_basis &basis) { return basis.size(-1); }

template <>
deg_t
basis_info<lal::tensor_basis>::native_degree(const lal::tensor_basis &basis, const this_key_type &key) {
    return static_cast<deg_t>(key.degree());
}
template <>
deg_t
basis_info<lal::tensor_basis>::degree(const lal::tensor_basis &basis, esig::key_type key) {
    return native_degree(basis, convert_key(basis, key));
}


template <>
typename lal::hall_basis::key_type
basis_info<lal::hall_basis>::convert_key(const lal::hall_basis &basis, esig::key_type key)
{
    return basis.index_to_key(key-1);
}
template <>
esig::key_type
basis_info<lal::hall_basis>::convert_key(const lal::hall_basis &basis, const this_key_type &key) {
    return 1 + basis.key_to_index(key);
}
template <>
esig::key_type
basis_info<lal::hall_basis>::first_key(const lal::hall_basis &basis) { return 1; }
template <>
esig::key_type
basis_info<lal::hall_basis>::last_key(const lal::hall_basis &basis) { return basis.size(-1) + 1; }
template <>
deg_t
basis_info<lal::hall_basis>::native_degree(const lal::hall_basis &basis, const this_key_type &key) {
    return static_cast<deg_t>(key.degree());
}
template <>
deg_t
basis_info<lal::hall_basis>::degree(const lal::hall_basis &basis, esig::key_type key) {
    return native_degree(basis, convert_key(basis, key));
}


template <typename MapType, typename Iterator>
struct iterator_traits<lal::dtl::sparse_iterator<MapType, Iterator>>
{
    using iterator = lal::dtl::sparse_iterator<MapType, Iterator>;

    static auto key(iterator it) noexcept -> decltype(it->key())
    {
        return it->key();
    }
    static auto value(iterator it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};

template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_traits<lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>>
{
    using iterator = lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>;
    static auto key(iterator it) noexcept -> decltype(it->key())
    {
        return it->key();
    }
    static auto value(iterator it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};
template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_traits<lal::dtl::dense_vector_const_iterator<Basis, Coefficients, Iterator>>
{
    using iterator = lal::dtl::dense_vector_const_iterator<Basis, Coefficients, Iterator>;
    static auto key(iterator it) noexcept -> decltype(it->key())
    {
        return it->key();
    }
    static auto value(iterator it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};



}


} // namespace algebra



}// namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_LITE_CONTEXT_H_
