//
// Created by user on 20/03/2022.
//

#ifndef ESIG_ALGEBRA_FALLBACK_CONTEXT_H_
#define ESIG_ALGEBRA_FALLBACK_CONTEXT_H_

#include <memory>
#include <unordered_map>

#include "lie_basis/lie_basis.h"
#include "fallback_maps.h"
#include "dense_lie.h"
#include "dense_tensor.h"
#include "sparse_lie.h"
#include "sparse_tensor.h"
#include "lie_basis/hall_extension.h"
#include <esig/algebra/context.h>
#include "detail/converting_lie_iterator_adaptor.h"

namespace esig {
namespace algebra {

namespace dtl {

template<typename Scalar, vector_type>
struct vector_selector;

}// namespace dtl

template<typename Scalar, vector_type VType>
using ftensor_type = typename dtl::vector_selector<Scalar, VType>::free_tensor;

template<typename Scalar, vector_type VType>
using lie_type = typename dtl::vector_selector<Scalar, VType>::lie;

std::shared_ptr<const context> get_fallback_context(deg_t width, deg_t depth, const scalars::scalar_type* ctype);

template<typename Scalar>
class fallback_context
    : public context {
    deg_t m_width;
    deg_t m_depth;
    std::shared_ptr<const tensor_basis> m_tensor_basis;
    std::shared_ptr<const lie_basis> m_lie_basis;
    using scalar_type = Scalar;

    maps m_maps;

    template<vector_type VType>
    lie_type<Scalar, VType> tensor_to_lie_impl(const free_tensor &arg) const;

    template<vector_type VType>
    lie_type<Scalar, VType> tensor_to_lie_impl(const ftensor_type<Scalar, VType> &arg) const;

    template<vector_type VType>
    ftensor_type<Scalar, VType> lie_to_tensor_impl(const lie &arg) const;

    template<vector_type VType>
    ftensor_type<Scalar, VType> lie_to_tensor_impl(const lie_type<Scalar, VType> &arg) const;

    template<vector_type VType>
    lie_type<Scalar, VType> cbh_impl(const std::vector<lie> &lies) const;

    template<vector_type VType>
    ftensor_type<Scalar, VType> compute_signature(const signature_data& data) const;

    template<typename Lie>
    Lie ad_x_n(deg_t d, const Lie &x, const Lie &y) const;

    template<typename Lie>
    Lie derive_series_compute(const Lie &incr, const Lie &pert) const;

    template<vector_type VType>
    ftensor_type<Scalar, VType> single_sig_derivative(
        const lie_type<Scalar, VType> &incr,
        const ftensor_type<Scalar, VType> &sig,
        const lie_type<Scalar, VType> &perturb) const;

    template<vector_type VType>
    ftensor_type<Scalar, VType>
    sig_derivative_impl(const std::vector<derivative_compute_info> &info) const;

    template<typename VectorImpl, template<typename> class VectorImplWrapper, typename VectorWrapper, typename Basis>
    VectorWrapper convert_impl(const VectorWrapper &arg, Basis basis) const;

public:
    fallback_context(deg_t width, deg_t depth);

    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;

    //    const algebra_basis &borrow_tbasis() const noexcept override;
    //    const algebra_basis &borrow_lbasis() const noexcept override;

    //    std::shared_ptr<const tensor_basis> tbasis() const noexcept;
    //    std::shared_ptr<const lie_basis> lbasis() const noexcept;

    basis get_tensor_basis() const override;
    basis get_lie_basis() const override;

    std::shared_ptr<const context> get_alike(deg_t new_depth) const override;
    std::shared_ptr<const context> get_alike(const scalars::scalar_type* new_coeff) const override;
    std::shared_ptr<const context> get_alike(deg_t new_depth, const scalars::scalar_type* new_coeff) const override;
    std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, const scalars::scalar_type* new_coeff) const override;


    free_tensor convert(const free_tensor &arg, vector_type new_vec_type) const override;
    lie convert(const lie &arg, vector_type new_vec_type) const override;

    free_tensor zero_tensor(vector_type vtype) const override;
    lie zero_lie(vector_type vtype) const override;
    lie cbh(const std::vector<lie> &lies, vector_type vtype) const override;
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const override;
    dimn_t lie_size(deg_t d) const noexcept override;
    dimn_t tensor_size(deg_t d) const noexcept override;

    free_tensor construct_tensor(const vector_construction_data &data) const override;
    lie construct_lie(const vector_construction_data &data) const override;
    free_tensor lie_to_tensor(const lie &arg) const override;
    lie tensor_to_lie(const free_tensor &arg) const override;
    free_tensor to_signature(const lie &log_sig) const override;
    free_tensor signature(const signature_data& data) const override;
    lie log_signature(const signature_data& data) const override;
};

class fallback_context_maker : public context_maker {
public:
    std::shared_ptr<const context> get_context(deg_t width, deg_t depth, const scalars::scalar_type* ctype) const override;
    bool can_get(deg_t width, deg_t depth, const scalars::scalar_type* ctype) const noexcept override;
    int get_priority(const std::vector<std::string> &preferences) const noexcept override;
};

namespace dtl {

template<typename Scalar>
struct vector_selector<Scalar, vector_type::sparse> {
    using free_tensor = sparse_tensor<Scalar>;
    using lie = sparse_lie<Scalar>;
};

template<typename Scalar>
struct vector_selector<Scalar, vector_type::dense> {
    using free_tensor = dense_tensor<Scalar>;
    using lie = dense_lie<Scalar>;
};


}// namespace dtl

template<typename Scalar>
template<vector_type VType>
lie_type<Scalar, VType> fallback_context<Scalar>::cbh_impl(const std::vector<lie> &lies) const {
    using lie_t = lie_type<Scalar, VType>;
    dtl::converting_lie_iterator_adaptor<lie_t> begin(m_lie_basis, lies.begin());
    dtl::converting_lie_iterator_adaptor<lie_t> end(m_lie_basis, lies.end());
    return m_maps.cbh(begin, end);
}

template<typename Scalar>
template<vector_type VType>
ftensor_type<Scalar, VType> fallback_context<Scalar>::lie_to_tensor_impl(const lie &arg) const {
    const auto &inner = algebra_access<lie_interface>::get<lie_type<Scalar, VType>>(arg);
    return lie_to_tensor_impl<VType>(inner);
}
template<typename Scalar>
template<vector_type VType>
ftensor_type<Scalar, VType> fallback_context<Scalar>::lie_to_tensor_impl(const lie_type<Scalar, VType> &arg) const {
    return m_maps.template lie_to_tensor<lie_type<Scalar, VType>, ftensor_type<Scalar, VType>>(arg);
}
template<typename Scalar>
template<vector_type VType>
lie_type<Scalar, VType> fallback_context<Scalar>::tensor_to_lie_impl(const free_tensor &arg) const {
    //    try {
    const auto &inner = algebra_access<free_tensor_interface>::get<ftensor_type<Scalar, VType>>(arg);
    return tensor_to_lie_impl<VType>(inner);
    //    } catch (std::bad_cast&) {
    //
    //    }
}
template<typename Scalar>
template<vector_type VType>
lie_type<Scalar, VType> fallback_context<Scalar>::tensor_to_lie_impl(const ftensor_type<Scalar, VType> &arg) const {
    return m_maps.template tensor_to_lie<lie_type<Scalar, VType>, ftensor_type<Scalar, VType>>(arg);
}

namespace dtl {

template<typename Vector, typename Basis>
Vector construct_vector(
    std::shared_ptr<Basis> basis,
    const esig::algebra::vector_construction_data& data) {
    using scalar_type = typename algebra_info<Vector>::scalar_type;

    if (data.data.is_null()) {
        return Vector(std::move(basis));
    }

    const auto size = data.data.size();
    const void *data_begin = data.data.ptr();
    std::vector<scalar_type> tmp;
    if (data.data.type() != ::esig::scalars::dtl::scalar_type_holder<scalar_type>::get_type()) {
        tmp.resize(data.data.size());
        data.data.type()->convert_copy(tmp.data(), data.data, size);
        data_begin = tmp.data();
    }

    const auto *s_data_begin = static_cast<const scalar_type *>(data_begin);

    if (data.key_data == nullptr) {
        // Construct from densely defined data
        const auto *s_data_end = s_data_begin + size;
        return Vector(std::move(basis), s_data_begin, s_data_end);
    }

    const auto* keys = data.key_data;

    Vector result(std::move(basis));
    for (dimn_t i=0; i<size; ++i) {
        result.add_scal_prod(keys[i], s_data_begin[i]);
    }

    return result;
}

}// namespace dtl

template<typename Scalar>
template<vector_type VType>
ftensor_type<Scalar, VType> fallback_context<Scalar>::compute_signature(const signature_data& data) const {
    //TODO: In the future, use a method on signature_data to get the
    // correct depth for the increments.
    ftensor_type<Scalar, VType> result(m_tensor_basis);
    const auto nrows = data.data_stream.row_count();

    for (dimn_t i=0; i<nrows; ++i) {
        auto row = data.data_stream[i];
        const auto* keys = data.key_stream.empty() ? nullptr : data.key_stream[i];
        vector_construction_data row_cdata { row, keys, VType };

        // This if statement is known at compile-time, so the optimizer should
        // go ahead and throw away the branch that does not apply.
        auto lie = dtl::construct_vector<lie_type<Scalar, VType>>(m_lie_basis, row_cdata);
        result.fmexp_inplace(m_maps.template lie_to_tensor<lie_type<Scalar, VType>, ftensor_type<Scalar, VType>>(lie));
    }

    return result;
}

template<typename Scalar>
template<typename Lie>
Lie fallback_context<Scalar>::ad_x_n(deg_t d, const Lie &x, const Lie &y) const {
    auto tmp = x * y;// [x, y]
    while (--d) {
        tmp = x * tmp;// x*tmp is [x, tmp] here
    }
    return tmp;
}

template<typename Scalar>
template<typename Lie>
Lie fallback_context<Scalar>::derive_series_compute(const Lie &incr, const Lie &pert) const {
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

template<typename Scalar>
template<vector_type VType>
ftensor_type<Scalar, VType> fallback_context<Scalar>::single_sig_derivative(
    const lie_type<Scalar, VType> &incr,
    const ftensor_type<Scalar, VType> &sig,
    const lie_type<Scalar, VType> &perturb) const {
    auto derive_ser = lie_to_tensor_impl<VType>(derive_series_compute(incr, perturb));

    return sig * derive_ser;
}

template<typename Scalar>
template<vector_type VType>
ftensor_type<Scalar, VType> fallback_context<Scalar>::sig_derivative_impl(const std::vector<derivative_compute_info> &info) const {
    if (info.empty()) {
        return ftensor_type<Scalar, VType>(m_tensor_basis);
    }

    ftensor_type<Scalar, VType> result(m_tensor_basis);

    using access = algebra_access<lie_interface>;

    for (const auto &data : info) {
        const auto &perturb = access::get<lie_type<Scalar, VType>>(data.perturbation);
        const auto &incr = access::get<lie_type<Scalar, VType>>(data.logsig_of_interval);
        auto sig = exp(lie_to_tensor_impl<VType>(incr));

        result *= sig;
        result += single_sig_derivative<VType>(incr, sig, perturb);
    }

    return result;
}

template<typename Scalar>
template<typename VectorImpl, template<typename> class VectorImplWrapper, typename VectorWrapper, typename Basis>
VectorWrapper fallback_context<Scalar>::convert_impl(const VectorWrapper &arg, Basis basis) const {
    const auto *base = algebra_access<typename VectorImplWrapper<VectorImpl>::interface_t>::get(arg);
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
            using access = algebra_access<typename VectorImplWrapper<VectorImpl>::interface_t>;
            const auto &impl = access::template get<VectorImpl>(*base);
            if (arg.coeff_type() == ctype()) {
                result.assign(impl.data());
            } else {
                const auto &data = impl.data();
                result.assign(data.begin(), data.end());
            }

        } else {
            /*
             * Would it be better to have some more sophisticated mechanism for copying data
             * across when it is possible. Using the vector iterator is really not a good idea
             * because every iteration involves a heap allocation. Think on this more.
             */
            for (const auto &it : *base) {
                result.add_scal_prod(it.key(), scalars::scalar_cast<Scalar>(it.value()));
            }
        }
    }
    return VectorWrapper(std::move(result), this);
}

template<typename Scalar>
fallback_context<Scalar>::fallback_context(deg_t width, deg_t depth)
    : context(::esig::scalars::dtl::scalar_type_holder<Scalar>::get_type()),
      m_width(width), m_depth(depth),
      m_tensor_basis(new tensor_basis(width, depth)),
      m_lie_basis(new lie_basis(width, depth)),
      m_maps(m_tensor_basis, m_lie_basis) {}


template<typename Scalar>
free_tensor fallback_context<Scalar>::zero_tensor(vector_type vtype) const {
    return context::zero_tensor(vtype);
}
template<typename Scalar>
lie fallback_context<Scalar>::zero_lie(vector_type vtype) const {
    return context::zero_lie(vtype);
}
template<typename Scalar>
deg_t fallback_context<Scalar>::width() const noexcept {
    return m_width;
}
template<typename Scalar>
deg_t fallback_context<Scalar>::depth() const noexcept {
    return m_depth;
}

//template<coefficient_type Scalar>
//const algebra_basis &fallback_context<Scalar>::borrow_tbasis() const noexcept {
//    return *m_tensor_basis;
//}
//template<coefficient_type Scalar>
//const algebra_basis &fallback_context<Scalar>::borrow_lbasis() const noexcept {
//    return *m_lie_basis;
//}
//template<coefficient_type Scalar>
//std::shared_ptr<tensor_basis> fallback_context<Scalar>::tbasis() const noexcept {
//    return m_tensor_basis;
//}
//template<coefficient_type Scalar>
//std::shared_ptr<lie_basis> fallback_context<Scalar>::lbasis() const noexcept {
//    return m_lie_basis;
//
//}

template<typename Scalar>
std::shared_ptr<const context> fallback_context<Scalar>::get_alike(deg_t new_depth) const {
    return get_fallback_context(m_width, new_depth, ctype());
}
template<typename Scalar>
std::shared_ptr<const context> fallback_context<Scalar>::get_alike(const scalars::scalar_type* new_coeff) const {
    return get_fallback_context(m_width, m_depth, new_coeff);
}
template<typename Scalar>
std::shared_ptr<const context> fallback_context<Scalar>::get_alike(deg_t new_depth, const scalars::scalar_type* new_coeff) const {
    return get_fallback_context(m_width, new_depth, new_coeff);
}
template<typename Scalar>
std::shared_ptr<const context> fallback_context<Scalar>::get_alike(deg_t new_width, deg_t new_depth, const scalars::scalar_type* new_coeff) const {
    return get_fallback_context(new_width, new_depth, new_coeff);
}

template<typename Scalar>
free_tensor fallback_context<Scalar>::convert(const free_tensor &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) convert_impl<ftensor_type<Scalar, (VTYPE)>, dtl::free_tensor_implementation>(arg, m_tensor_basis)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
lie fallback_context<Scalar>::convert(const lie &arg, vector_type new_vec_type) const {
#define ESIG_SWITCH_FN(VTYPE) convert_impl<lie_type<Scalar, (VTYPE)>, dtl::lie_implementation>(arg, m_lie_basis)
    ESIG_MAKE_VTYPE_SWITCH(new_vec_type)
#undef ESIG_SWITCH_FN
}

template<typename Scalar>
lie fallback_context<Scalar>::cbh(const std::vector<lie> &lies, vector_type vtype) const {
#define ESIG_SWITCH_FN(VTYPE) lie(cbh_impl<VTYPE>(lies), this)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
free_tensor fallback_context<Scalar>::sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(sig_derivative_impl<VTYPE>(info), this)
    ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
dimn_t fallback_context<Scalar>::lie_size(deg_t d) const noexcept {
    return m_lie_basis->size(static_cast<int>(d));
}
template<typename Scalar>
dimn_t fallback_context<Scalar>::tensor_size(deg_t d) const noexcept {
    return m_tensor_basis->size(static_cast<int>(d));
}
//template<coefficient_type Scalar>
//std::shared_ptr<tensor_basis> fallback_context<Scalar>::get_tensor_basis() const noexcept {
//    return m_tensor_basis;
//}
//template<coefficient_type Scalar>
//std::shared_ptr<lie_basis> fallback_context<Scalar>::get_lie_basis() const noexcept {
//    return m_lie_basis;
//}
template<typename Scalar>
free_tensor fallback_context<Scalar>::construct_tensor(const vector_construction_data &data) const {
    switch (data.vect_type) {
        case vector_type::dense:
            return free_tensor(dtl::construct_vector<ftensor_type<Scalar, vector_type::dense>>(m_tensor_basis, data), this);
        case vector_type::sparse:
            return free_tensor(dtl::construct_vector<ftensor_type<Scalar, vector_type::sparse>>(m_tensor_basis, data), this);
    }
    throw std::invalid_argument("invalid vector type");
}
template<typename    Scalar>
lie fallback_context<Scalar>::construct_lie(const vector_construction_data &data) const {
    switch (data.vect_type) {
        case vector_type::dense:
            return lie(dtl::construct_vector<lie_type<Scalar, vector_type::dense>>(m_lie_basis, data), this);
        case vector_type::sparse:
            return lie(dtl::construct_vector<lie_type<Scalar, vector_type::sparse>>(m_lie_basis, data), this);
    }
    throw std::invalid_argument("invalid vector type");
}
template<typename Scalar>
free_tensor fallback_context<Scalar>::lie_to_tensor(const lie &arg) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(lie_to_tensor_impl<(VTYPE)>(arg), this)
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
}

template<typename Scalar>
lie fallback_context<Scalar>::tensor_to_lie(const free_tensor &arg) const {
#define ESIG_SWITCH_FN(VTYPE) lie(tensor_to_lie_impl<(VTYPE)>(arg), this)
    ESIG_MAKE_VTYPE_SWITCH(arg.storage_type());
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
free_tensor fallback_context<Scalar>::to_signature(const lie &log_sig) const {
    return lie_to_tensor(log_sig).exp();
}
template<typename Scalar>
free_tensor fallback_context<Scalar>::signature(const signature_data& data) const {
#define ESIG_SWITCH_FN(VTYPE) free_tensor(exp(log(compute_signature<(VTYPE)>(std::move(data)))), this);
    ESIG_MAKE_VTYPE_SWITCH(data.vect_type)
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
lie fallback_context<Scalar>::log_signature(const signature_data& data) const {
#define ESIG_SWITCH_FN(VTYPE) lie(tensor_to_lie_impl<(VTYPE)>(log(compute_signature<(VTYPE)>(std::move(data)))), this)
    ESIG_MAKE_VTYPE_SWITCH(data.vect_type)
#undef ESIG_SWITCH_FN
}
template<typename Scalar>
basis fallback_context<Scalar>::get_tensor_basis() const {
    return basis(m_tensor_basis);
}
template<typename Scalar>
basis fallback_context<Scalar>::get_lie_basis() const {
    return basis(m_lie_basis);
}

}// namespace algebra
}// namespace esig

#endif//ESIG_ALGEBRA_FALLBACK_CONTEXT_H_
