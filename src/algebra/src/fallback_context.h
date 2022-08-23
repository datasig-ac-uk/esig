//
// Created by user on 20/03/2022.
//

#ifndef ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_
#define ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_

#include <esig/algebra/context.h>
#include "dense_tensor.h"
#include "dense_lie.h"
#include "sparse_tensor.h"
#include "sparse_lie.h"
#include "lie_basis/lie_basis.h"
#include "lie_basis/hall_extension.h"
#include "fallback_maps.h"
#include "detail/converting_lie_iterator_adaptor.h"
#include <memory>
#include <unordered_map>


namespace esig {
namespace algebra {

namespace dtl {

template<coefficient_type, vector_type>
struct vector_selector;

} // namespace dtl

template <coefficient_type CType, vector_type VType>
using ftensor_type = typename dtl::vector_selector<CType, VType>::free_tensor;

template <coefficient_type CType, vector_type VType>
using lie_type = typename dtl::vector_selector<CType, VType>::lie;

template<typename Lie>
struct to_lie_helper {
    using scalar_type = typename Lie::scalar_type;

    to_lie_helper(std::shared_ptr<lie_basis> b, deg_t degree)
        : m_basis(std::move(b)), m_degree(degree) {}

    Lie operator()(const char *b, const char *e) const {
        return {m_basis,
                reinterpret_cast<const scalar_type *>(b),
                reinterpret_cast<const scalar_type *>(e)};
    }

    Lie operator()(data_iterator *data) const {
        Lie result(m_basis);
        while (data->next_sparse()) {
            auto kv = reinterpret_cast<const std::pair<key_type, scalar_type> *>(data->sparse_kv_pair());
            result[kv->first] = kv->second;
        }
        return result;
    }

private:
    std::shared_ptr<lie_basis> m_basis;
    deg_t m_degree;
};



std::shared_ptr<context> get_fallback_context(deg_t width, deg_t depth, coefficient_type ctype);



template <coefficient_type CType>
class fallback_context
    : public context
{
    deg_t m_width;
    deg_t m_depth;
    std::shared_ptr<tensor_basis> m_tensor_basis;
    std::shared_ptr<lie_basis> m_lie_basis;

    std::shared_ptr<data_allocator> m_coeff_alloc;
    std::shared_ptr<data_allocator> m_pair_alloc;

    using scalar_type = type_of_coeff<CType>;

    maps m_maps;

    template < vector_type VType>
    lie_type<CType, VType> tensor_to_lie_impl(const free_tensor& arg) const;

    template <vector_type VType>
    lie_type<CType, VType> tensor_to_lie_impl(const ftensor_type<CType, VType>& arg) const;

    template < vector_type VType>
    ftensor_type<CType, VType> lie_to_tensor_impl(const lie& arg) const;

    template <vector_type VType>
    ftensor_type<CType, VType> lie_to_tensor_impl(const lie_type<CType, VType>& arg) const;

    template < vector_type VType>
    lie_type<CType, VType> cbh_impl(const std::vector<lie>& lies) const;

    template< vector_type VType>
    ftensor_type<CType, VType> compute_signature(signature_data data) const;

    template<typename Lie>
    Lie ad_x_n(deg_t d, const Lie &x, const Lie &y) const;

    template<typename Lie>
    Lie derive_series_compute(const Lie &incr, const Lie &pert) const;

    template< vector_type VType>
    ftensor_type<CType, VType> single_sig_derivative(
        const lie_type<CType, VType> &incr,
        const ftensor_type<CType, VType> &sig,
        const lie_type<CType, VType> &perturb
        ) const;

    template< vector_type VType>
    ftensor_type<CType, VType>
    sig_derivative_impl(const std::vector<derivative_compute_info> &info) const;

    template<typename VectorImpl, template <typename> class VectorImplWrapper, typename VectorWrapper, typename Basis>
    VectorWrapper convert_impl(const VectorWrapper &arg, Basis basis) const;

public:

    fallback_context(deg_t width, deg_t depth);


    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    coefficient_type ctype() const noexcept override;

    const algebra_basis &borrow_tbasis() const noexcept override;
    const algebra_basis &borrow_lbasis() const noexcept override;

    std::shared_ptr<tensor_basis> tbasis() const noexcept;
    std::shared_ptr<lie_basis> lbasis() const noexcept;

    std::shared_ptr<context> get_alike(deg_t new_depth) const override;
    std::shared_ptr<context> get_alike(coefficient_type new_coeff) const override;
    std::shared_ptr<context> get_alike(deg_t new_depth, coefficient_type new_coeff) const override;
    std::shared_ptr<context> get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const override;

    std::shared_ptr<data_allocator> coefficient_alloc() const override;
    std::shared_ptr<data_allocator> pair_alloc() const override;

    free_tensor convert(const free_tensor &arg, vector_type new_vec_type) const override;
    lie convert(const lie &arg, vector_type new_vec_type) const override;

    free_tensor zero_tensor(vector_type vtype) const override;
    lie zero_lie(vector_type vtype) const override;
    lie cbh(const std::vector<lie> &lies, vector_type vtype) const override;
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const override;
    dimn_t lie_size(deg_t d) const noexcept override;
    dimn_t tensor_size(deg_t d) const noexcept override;
    std::shared_ptr<tensor_basis> get_tensor_basis() const noexcept;
    std::shared_ptr<lie_basis> get_lie_basis() const noexcept;
    free_tensor construct_tensor(const vector_construction_data &data) const override;
    lie construct_lie(const vector_construction_data &data) const override;
    free_tensor lie_to_tensor(const lie &arg) const override;
    lie tensor_to_lie(const free_tensor &arg) const override;
    free_tensor to_signature(const lie &log_sig) const override;
    free_tensor signature(signature_data data) const override;
    lie log_signature(signature_data data) const override;

};


class fallback_context_maker : public context_maker
{
public:
    std::shared_ptr<context> get_context(deg_t width, deg_t depth, coefficient_type ctype) const override;
    bool can_get(deg_t width, deg_t depth, coefficient_type ctype) const noexcept override;
    int get_priority(const std::vector<std::string> &preferences) const noexcept override;
};



namespace dtl {

template<>
struct vector_selector<coefficient_type::sp_real, vector_type::sparse> {
    using free_tensor = sparse_tensor<float>;
    using lie = sparse_lie<float>;
};

template<>
struct vector_selector<coefficient_type::sp_real, vector_type::dense> {
    using free_tensor = dense_tensor<float>;
    using lie = dense_lie<float>;
};

template<>
struct vector_selector<coefficient_type::dp_real, vector_type::sparse> {
    using free_tensor = sparse_tensor<double>;
    using lie = sparse_lie<double>;
};

template<>
struct vector_selector<coefficient_type::dp_real, vector_type::dense> {
    using free_tensor = dense_tensor<double>;
    using lie = dense_lie<double>;
};
} // namespace dtl

template <coefficient_type CType>
template< vector_type VType>
lie_type<CType, VType> fallback_context<CType>::cbh_impl(const std::vector<lie> &lies) const {
    using lie_t = lie_type<CType, VType>;
    dtl::converting_lie_iterator_adaptor<lie_t> begin(m_lie_basis, lies.begin());
    dtl::converting_lie_iterator_adaptor<lie_t> end(m_lie_basis, lies.end());
    return m_maps.cbh(begin, end);
}

template <coefficient_type CType>
template<vector_type VType>
ftensor_type<CType, VType> fallback_context<CType>::lie_to_tensor_impl(const lie &arg) const
{
    const auto& inner = lie_base_access::get<lie_type<CType, VType>>(arg);
    return lie_to_tensor_impl<VType>(inner);
}
template<coefficient_type CType>
template<vector_type VType>
ftensor_type<CType, VType> fallback_context<CType>::lie_to_tensor_impl(const lie_type<CType, VType> &arg) const {
    return m_maps.template lie_to_tensor<lie_type<CType, VType>, ftensor_type<CType, VType>>(arg);
}
template <coefficient_type CType>
template<vector_type VType>
lie_type<CType, VType> fallback_context<CType>::tensor_to_lie_impl(const free_tensor &arg) const
{
//    try {
        const auto& inner = free_tensor_base_access::get<ftensor_type<CType, VType>>(arg);
        return tensor_to_lie_impl<VType>(inner);
//    } catch (std::bad_cast&) {
//
//    }
}
template <coefficient_type CType>
template <vector_type VType>
lie_type<CType, VType> fallback_context<CType>::tensor_to_lie_impl(const ftensor_type<CType, VType>& arg) const
{
    return m_maps.template tensor_to_lie<lie_type<CType, VType>, ftensor_type<CType, VType>>(arg);
}


template <coefficient_type CType>
template<vector_type VType>
ftensor_type<CType, VType> fallback_context<CType>::compute_signature(signature_data data) const {
    //TODO: In the future, use a method on signature_data to get the
    // correct depth for the increments.
    to_lie_helper<lie_type<CType, VType>> helper(get_lie_basis(), 1);
    ftensor_type<CType, VType> result(get_tensor_basis(), typename ftensor_type<CType, VType>::scalar_type(1));
    for (auto incr : data.template iter_increments(helper)) {
        result.fmexp_inplace(lie_to_tensor_impl<VType>(incr));
    }

    return result;
}

template <coefficient_type CType>
template<typename Lie>
Lie fallback_context<CType>::ad_x_n(deg_t d, const Lie &x, const Lie &y) const
{
    auto tmp = x * y;// [x, y]
    while (--d) {
        tmp = x * tmp;// x*tmp is [x, tmp] here
    }
    return tmp;
}

template <coefficient_type CType>
template<typename Lie>
Lie fallback_context<CType>::derive_series_compute(const Lie &incr, const Lie &pert) const
{
    Lie result(pert);

    typename Lie::scalar_type factor(1);
    for (deg_t d = 1; d <= m_depth; ++d) {
        factor *= static_cast<typename Lie::scalar_type>(d + 1);
        if (d % 2 == 0) {
            result.add_scal_div(ad_x_n(d, incr, pert), factor);
        } else {
            result.sub_scal_div(ad_x_n(d, incr, pert), factor);
        }
    }
    return result;
}

template <coefficient_type CType>
template<vector_type VType>
ftensor_type<CType, VType> fallback_context<CType>::single_sig_derivative(
    const lie_type<CType, VType> &incr,
    const ftensor_type<CType, VType> &sig,
    const lie_type<CType, VType> &perturb
    ) const
{
    auto derive_ser = lie_to_tensor_impl<VType>(derive_series_compute(incr, perturb));

    return sig * derive_ser;
}

template <coefficient_type CType>
template<vector_type VType>
ftensor_type<CType, VType> fallback_context<CType>::sig_derivative_impl(const std::vector<derivative_compute_info> &info) const
{
    if (info.empty()) {
        return ftensor_type<CType, VType>(get_tensor_basis());
    }

    ftensor_type<CType, VType> result(get_tensor_basis());

    for (const auto &data : info) {
        const auto &perturb = lie_base_access::get<lie_type<CType, VType>>(data.perturbation);
        const auto &incr = lie_base_access::get<lie_type<CType, VType>>(data.logsig_of_interval);
        auto sig = exp(lie_to_tensor_impl<VType>(incr));

        result *= sig;
        result += single_sig_derivative<VType>(incr, sig, perturb);
    }

    return result;
}

template<coefficient_type CType>
template<typename VectorImpl, template <typename> class VectorImplWrapper, typename VectorWrapper, typename Basis>
VectorWrapper fallback_context<CType>::convert_impl(const VectorWrapper &arg, Basis basis) const {
    const auto *base = algebra_base_access::get(arg);
    VectorImpl result(basis);

    if (base != nullptr) {
        if (arg.width() != m_width) {
            throw std::invalid_argument("vector width does not match");
        }

        if (typeid(*base) == typeid(VectorImplWrapper<VectorImpl>)) {
            /*
             * If the base is already of the same type as result then we can use assign/data
             * defined on the fallback vector types.
             */
            using access = algebra_implementation_access<VectorImplWrapper, VectorImpl>;
            const auto& impl = access::get(dynamic_cast<const VectorImplWrapper<VectorImpl> &>(*base));
            if (impl.coeff_type() == CType) {
                result.assign(impl.data());
            } else {
                const auto& data = impl.data();
                result.assign(data.begin(), data.end());
            }

        } else {
            /*
             * Would it be better to have some more sophisticated mechanism for copying data
             * across when it is possible. Using the vector iterator is really not a good idea
             * because every iteration involves a heap allocation. Think on this more.
             */
            for (const auto &it : *base) {
                result.add_scal_prod(it.key(), coefficient_cast<scalar_type>(it.value()));
            }
        }

    }
    return VectorWrapper(std::move(result), this);
}

template<coefficient_type CType>
fallback_context<CType>::fallback_context(deg_t width, deg_t depth)
    : m_width(width), m_depth(depth),
      m_tensor_basis(new tensor_basis(width, depth)),
      m_lie_basis(new lie_basis(width, depth)),
      m_coeff_alloc(allocator_for_coeff(CType)),
      m_pair_alloc(allocator_for_key_coeff(CType)),
      m_maps(m_tensor_basis, m_lie_basis)
{}



namespace dtl {

template <typename Scalar, template<typename> class Vector, typename Basis>
Vector<Scalar> construct_vector(
        std::shared_ptr<Basis> basis,
        esig::algebra::vector_construction_data data)
{
    if (data.input_type() == esig::algebra::input_data_type::value_array) {
        return {std::move(basis), reinterpret_cast<const Scalar *>(data.begin()), reinterpret_cast<const Scalar*>(data.end())};
    } else {
        // fill me in
        Vector<Scalar> result(std::move(basis));
        assert(data.item_size() == sizeof(std::pair<key_type, Scalar>));

        auto ptr = reinterpret_cast<const std::pair<key_type, Scalar>*>(data.begin());
        auto end = reinterpret_cast<const std::pair<key_type, Scalar>*>(data.end());
        for (; ptr != end; ++ptr) {
            result[ptr->first] = ptr->second;
        }

        return result;
    }
}


} // namespace dtl



template<coefficient_type CType>
free_tensor fallback_context<CType>::zero_tensor(vector_type vtype) const {
    return context::zero_tensor(vtype);
}
template<coefficient_type CType>
lie fallback_context<CType>::zero_lie(vector_type vtype) const {
    return context::zero_lie(vtype);
}
template<coefficient_type CType>
deg_t fallback_context<CType>::width() const noexcept {
    return m_width;
}
template<coefficient_type CType>
deg_t fallback_context<CType>::depth() const noexcept {
    return m_depth;
}
template<coefficient_type CType>
coefficient_type fallback_context<CType>::ctype() const noexcept {
    return coefficient_type::dp_real;
}
template<coefficient_type CType>
const algebra_basis &fallback_context<CType>::borrow_tbasis() const noexcept {
    return *m_tensor_basis;
}
template<coefficient_type CType>
const algebra_basis &fallback_context<CType>::borrow_lbasis() const noexcept {
    return *m_lie_basis;
}
template<coefficient_type CType>
std::shared_ptr<tensor_basis> fallback_context<CType>::tbasis() const noexcept {
    return m_tensor_basis;
}
template<coefficient_type CType>
std::shared_ptr<lie_basis> fallback_context<CType>::lbasis() const noexcept {
    return m_lie_basis;

}

template<coefficient_type CType>
std::shared_ptr<context> fallback_context<CType>::get_alike(deg_t new_depth) const {
    return get_fallback_context(m_width, new_depth, CType);
}
template<coefficient_type CType>
std::shared_ptr<context> fallback_context<CType>::get_alike(coefficient_type new_coeff) const {
    return get_fallback_context(m_width, m_depth, new_coeff);
}
template<coefficient_type CType>
std::shared_ptr<context> fallback_context<CType>::get_alike(deg_t new_depth, coefficient_type new_coeff) const {
    return get_fallback_context(m_width, new_depth, new_coeff);
}
template<coefficient_type CType>
std::shared_ptr<context> fallback_context<CType>::get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const {
    return get_fallback_context(new_width, new_depth, new_coeff);
}
template<coefficient_type CType>
std::shared_ptr<data_allocator> fallback_context<CType>::coefficient_alloc() const {
    return m_coeff_alloc;
}
template<coefficient_type CType>
std::shared_ptr<data_allocator> fallback_context<CType>::pair_alloc() const {
    return m_pair_alloc;
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::convert(const free_tensor &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) convert_impl<ftensor_type<CType, (VTYPE)>, dtl::free_tensor_implementation>(arg, m_tensor_basis)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}
template<coefficient_type CType>
lie fallback_context<CType>::convert(const lie &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) convert_impl<lie_type<CType, (VTYPE)>, dtl::lie_implementation>(arg, m_lie_basis)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}

template<coefficient_type CType>
lie fallback_context<CType>::cbh(const std::vector<lie> &lies, vector_type vtype) const {
#define ESIG_SWITCH_FN(VTYPE) lie(cbh_impl<VTYPE>(lies), this)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(sig_derivative_impl<VTYPE>(info), this)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}
template<coefficient_type CType>
dimn_t fallback_context<CType>::lie_size(deg_t d) const noexcept {
    return m_lie_basis->size(static_cast<int>(d));
}
template<coefficient_type CType>
dimn_t fallback_context<CType>::tensor_size(deg_t d) const noexcept {
    return m_tensor_basis->size(static_cast<int>(d));
}
template<coefficient_type CType>
std::shared_ptr<tensor_basis> fallback_context<CType>::get_tensor_basis() const noexcept {
    return m_tensor_basis;
}
template<coefficient_type CType>
std::shared_ptr<lie_basis> fallback_context<CType>::get_lie_basis() const noexcept {
    return m_lie_basis;
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::construct_tensor(const vector_construction_data &data) const {
    switch (data.vtype()) {
        case vector_type::dense:
            return free_tensor(dtl::construct_vector<scalar_type, dense_tensor>(m_tensor_basis, data), this);
        case vector_type::sparse:
            return free_tensor(dtl::construct_vector<scalar_type, sparse_tensor>(m_tensor_basis, data), this);
    }
    throw std::invalid_argument("invalid vector type");
}
template<coefficient_type CType>
lie fallback_context<CType>::construct_lie(const vector_construction_data &data) const {
    switch (data.vtype()) {
        case vector_type::dense:
            return lie(dtl::construct_vector<scalar_type, dense_lie>(m_lie_basis, data), this);
        case vector_type::sparse:
            return lie(dtl::construct_vector<scalar_type, sparse_lie>(m_lie_basis, data), this);
    }
    throw std::invalid_argument("invalid vector type");
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::lie_to_tensor(const lie &arg) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(lie_to_tensor_impl<(VTYPE)>(arg), this)
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
}


template<coefficient_type CType>
lie fallback_context<CType>::tensor_to_lie(const free_tensor &arg) const {
#define ESIG_SWITCH_FN(VTYPE) lie(tensor_to_lie_impl<(VTYPE)>(arg), this)
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type());
#undef ESIG_SWITCH_FN
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::to_signature(const lie &log_sig) const {
    return lie_to_tensor(log_sig).exp();
}
template<coefficient_type CType>
free_tensor fallback_context<CType>::signature(signature_data data) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(exp(log(compute_signature<(VTYPE)>(std::move(data)))), this);
    ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
}
template<coefficient_type CType>
lie fallback_context<CType>::log_signature(signature_data data) const {
#define ESIG_SWITCH_FN(VTYPE) lie(tensor_to_lie_impl<(VTYPE)>(log(compute_signature<(VTYPE)>(std::move(data)))), this)
    ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
}

}// namespace algebra
}// namespace esig


#endif//ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_
