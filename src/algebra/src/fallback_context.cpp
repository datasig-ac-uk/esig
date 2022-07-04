//
// Created by user on 20/03/2022.
//

#include "fallback_context.h"
#include "lie_basis/lie_basis.h"
#include "tensor_basis/tensor_basis.h"
#include <esig/implementation_types.h>
#include "fallback_maps_internal.h"

namespace esig {
namespace algebra {
namespace {

// Some helper functions

/**
 * @brief Get the degree from the size of an input vector
 * @param basis basis for the algebra_old to be constructed
 * @param size size of the input data
 * @return degree of data
 */
deg_t get_degree(const algebra_basis& basis, dimn_t size)
{
    //TODO: Replace with std algorithms?
    deg_t d;
    for (d = 0; d <= basis.depth(); ++d) {
        if (size < basis.size(static_cast<int>(d)))
            break;
    }
    return d;
}


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

std::vector<std::pair<key_type, int>> expand_func(let_t letter)
{
    return {{letter, 1}};
}



template <typename Lie>
struct to_lie_helper
{
    using scalar_type = typename Lie::scalar_type;

    to_lie_helper(std::shared_ptr<lie_basis> b, deg_t degree)
            : m_basis(std::move(b)), m_degree(degree)
    {}

    Lie operator()(const char* b, const char* e) const
    {
        return {m_basis,
                reinterpret_cast<const scalar_type*>(b),
                reinterpret_cast<const scalar_type*>(e)};
    }

    Lie operator()(data_iterator* data) const
    {
        Lie result(m_basis);
        while (data->next_sparse()) {
            auto kv = reinterpret_cast<const std::pair<key_type, scalar_type>*>(data->sparse_kv_pair());
            result[kv->first] = kv->second;
        }
        return result;
    }



private:
    std::shared_ptr<lie_basis> m_basis;
    deg_t m_degree;
};






template <coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> compute_signature(const fallback_context& ctx, signature_data data)
{
    //TODO: In the future, use a method on signature_data to get the
    // correct depth for the increments.
    to_lie_helper<lie_type<CType, VType>> helper(ctx.get_lie_basis(), 1);
    ftensor_type<CType, VType> result(ctx.get_tensor_basis(), typename ftensor_type<CType, VType>::scalar_type(1));
    for (auto incr : data.template iter_increments(helper)) {
        result.fmexp_inplace(lie_to_tensor_impl<CType, VType>(ctx, incr));
    }

    return result;
}

template<typename Lie>
Lie ad_x_n(deg_t d, const Lie &x, const Lie &y)
{
    auto tmp = x * y;// [x, y]
    while (--d) {
        tmp = x * tmp;// x*tmp is [x, tmp] here
    }
    return tmp;
}

template<typename Lie>
Lie derive_series_compute(const fallback_context &ctx, const Lie &incr, const Lie &pert)
{
    auto depth = ctx.depth();
    Lie result(ctx.get_lie_basis());

    typename Lie::scalar_type factor(1);
    for (deg_t d = 1; d <= depth; ++d) {
        factor /= static_cast<typename Lie::scalar_type>(d + 1);
        if (d & 1) {
            result.add_scal_div(ad_x_n(d, incr, pert), factor);
        } else {
            result.sub_scal_div(ad_x_n(d, incr, pert), factor);
        }
    }
    return result;
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> single_sig_derivative(
        const fallback_context &ctx,
        const lie_type<CType, VType> &incr,
        const ftensor_type<CType, VType> &sig,
        const lie_type<CType, VType> &perturb)
{
    return sig * lie_to_tensor_impl<CType, VType>(ctx, derive_series_compute(ctx, incr, perturb));
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> sig_derivative_impl(const fallback_context &ctx, const std::vector<derivative_compute_info> &info)
{
    if (info.empty()) {
        return ftensor_type<CType, VType>(ctx.get_tensor_basis());
    }

    ftensor_type<CType, VType> result(ctx.get_tensor_basis());

    for (const auto &data : info) {
        const auto &perturb = lie_base_access::get<lie_type<CType, VType>>(data.perturbation);
        const auto &incr = lie_base_access::get<lie_type<CType, VType>>(data.logsig_of_interval);
        auto sig = exp(lie_to_tensor_impl<CType, VType>(ctx, incr));

        result *= sig;
        result += single_sig_derivative<CType, VType>(ctx, incr, sig, perturb);
    }

    return result;
}
} // namespace


namespace dtl {

expand_binop::expand_binop(std::shared_ptr<tensor_basis> arg)
    : m_basis(std::move(arg))
{
}
expand_binop::result_t expand_binop::operator()(const expand_binop::result_t &a, const expand_binop::result_t &b) const
{
    const auto& powers = m_basis->powers();
    std::map<key_type, int> tmp;
    for (const auto& lhs : a) {
        for (const auto& rhs : b) {
            auto lhs_deg = m_basis->degree(lhs.first);
            auto rhs_deg = m_basis->degree(rhs.first);
            auto lhs_shift = powers[lhs_deg];
            auto rhs_shift = powers[rhs_deg];

            tmp[lhs.first*rhs_shift + rhs.first] += lhs.second*rhs.second;
            tmp[rhs.first*lhs_shift + lhs.first] -= rhs.second*lhs.second;
        }
    }
    return {tmp.begin(), tmp.end()};
}

bracketing_item::bracketing_item() : std::vector<std::pair<key_type, int>>(), b_set(false)
{
}
bracketing_item::operator bool() const noexcept
{
    return b_set;
}
void bracketing_item::set() noexcept
{
    b_set = true;
}

} // namespace dtl


static register_maker_helper<fallback_context_maker> fb_context_maker;



fallback_context::fallback_context(deg_t width, deg_t depth, coefficient_type ctype)
    : m_width(width), m_depth(depth), m_ctype(ctype),
      m_tensor_basis(new tensor_basis(width, depth)),
      m_lie_basis(new lie_basis(width, depth)),
      m_expand(m_lie_basis->get_hall_set(), expand_func, dtl::expand_binop(m_tensor_basis)),
      m_coeff_alloc(allocator_for_coeff(ctype)),
      m_pair_alloc(allocator_for_key_coeff(ctype))
{
}


deg_t fallback_context::width() const noexcept
{
    return m_width;
}
deg_t fallback_context::depth() const noexcept
{
    return m_depth;
}
dimn_t fallback_context::lie_size(deg_t d) const noexcept
{
    return m_lie_basis->size(static_cast<int>(d));
}
dimn_t fallback_context::tensor_size(deg_t d) const noexcept
{
    return m_tensor_basis->size(static_cast<int>(d));
}
std::shared_ptr<tensor_basis> fallback_context::get_tensor_basis() const noexcept
{
    return m_tensor_basis;
}
std::shared_ptr<lie_basis> fallback_context::get_lie_basis() const noexcept
{
    return m_lie_basis;
}


free_tensor fallback_context::construct_tensor(const vector_construction_data &data) const
{
    assert(m_ctype == data.ctype());
    switch(m_ctype) {
        case coefficient_type::dp_real:
            switch (data.vtype()) {
                case vector_type::dense:
                    return free_tensor(construct_vector<double, dense_tensor>(m_tensor_basis, data), this);
                case vector_type::sparse:
                    return free_tensor(construct_vector<double, sparse_tensor>(m_tensor_basis, data), this);
            }
        case coefficient_type::sp_real:
            switch(data.vtype()) {
                case vector_type::dense:
                    return free_tensor(construct_vector<float, dense_tensor>(m_tensor_basis, data), this);
                case vector_type::sparse:
                    return free_tensor(construct_vector<float, sparse_tensor>(m_tensor_basis, data), this);
            }
    }
    throw std::invalid_argument("invalid construction data");
}
lie fallback_context::construct_lie(const vector_construction_data &data) const
{
    assert(m_ctype == data.ctype());
    switch (m_ctype) {
        case coefficient_type::dp_real:
            switch (data.vtype()) {
                case vector_type::dense:
                    return lie(construct_vector<double, dense_lie>(m_lie_basis, data), this);
                case vector_type::sparse:
                    return lie(construct_vector<double, sparse_lie>(m_lie_basis, data), this);
            }
        case coefficient_type::sp_real:
            switch (data.vtype()) {
                case vector_type::dense:
                    return lie(construct_vector<float, dense_lie>(m_lie_basis, data), this);
                case vector_type::sparse:
                    return lie(construct_vector<float, sparse_lie>(m_lie_basis, data), this);
            }
    }
    throw std::invalid_argument("invalid construction data");
}
free_tensor fallback_context::to_signature(const lie &log_sig) const
{
    return lie_to_tensor(log_sig).exp();
}
free_tensor fallback_context::signature(signature_data data) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(exp(log(compute_signature<CTYPE, VTYPE>(*this, std::move(data)))), this)
ESIG_MAKE_SWITCH(m_ctype, data.vtype())
#undef ESIG_SWITCH_FN
//    switch (data.ctype()) {
//        case coefficient_type::sp_real:
//            switch (data.vtype()){
//                case vector_type::sparse:
//                    return free_tensor(exp(log(compute_signature<coefficient_type::sp_real, vector_type::sparse>(*this, std::move(data)))));
//                case vector_type::dense:
//                    return free_tensor(exp(log(compute_signature<coefficient_type::sp_real, vector_type::dense>(*this, std::move(data)))));
//            }
//        case coefficient_type::dp_real:
//            switch (data.vtype()) {
//                case vector_type::sparse:
//                    return free_tensor(exp(log(compute_signature<coefficient_type::dp_real, vector_type::sparse>(*this, std::move(data)))));
//                case vector_type::dense:
//                    return free_tensor(exp(log(compute_signature<coefficient_type::dp_real, vector_type::dense>(*this, std::move(data)))));
//            }
//    }
//    throw std::invalid_argument("bad construction data");
}
lie fallback_context::log_signature(signature_data data) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(tensor_to_lie_impl<CTYPE, VTYPE>(*this, log(compute_signature<CTYPE, VTYPE>(*this, std::move(data)))), this)
ESIG_MAKE_SWITCH(m_ctype, data.vtype())
#undef ESIG_SWITCH_FN
//    switch (data.ctype()) {
//        case coefficient_type::sp_real:
//            switch (data.vtype()) {
//                case vector_type::sparse:
//                    return lie(tensor_to_lie_impl<>(*this, log(compute_signature<ss_lie_type, ss_free_tensor_type>(*this, std::move(data)))));
//                case vector_type::dense:
//                    return lie(tensor_to_lie_impl<sd_lie_type>(*this, log(compute_signature<sd_lie_type, sd_free_tensor_type>(*this, std::move(data)))));
//            }
//        case coefficient_type::dp_real:
//            switch (data.vtype()) {
//                case vector_type::sparse:
//                    return lie(tensor_to_lie_impl<ds_lie_type>(*this, log(compute_signature<ds_lie_type, ds_free_tensor_type>(*this, std::move(data)))));
//                case vector_type::dense:
//                    return lie(tensor_to_lie_impl<dd_lie_type>(*this, log(compute_signature<dd_lie_type, dd_free_tensor_type>(*this, std::move(data)))));
//            }
//    }
//    throw std::invalid_argument("bad construction data");
}
free_tensor fallback_context::sig_derivative(
        const std::vector<derivative_compute_info> &info,
        vector_type vtype,
        vector_type
        ) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(sig_derivative_impl<CTYPE, VTYPE>(*this, info), this)
ESIG_MAKE_SWITCH(m_ctype, vtype)
#undef ESIG_SWITCH_FN
//    switch (m_ctype) {
//        case coefficient_type::sp_real:
//            switch (vtype) {
//                case vector_type::sparse:
//                    return free_tensor(sig_derivative_impl<ss_lie_type, ss_free_tensor_type>(*this, info));
//                case vector_type::dense:
//                    return free_tensor(sig_derivative_impl<sd_lie_type, sd_free_tensor_type>(*this, info));
//            }
//        case coefficient_type::dp_real:
//            switch (vtype) {
//                case vector_type::sparse:
//                    return free_tensor(sig_derivative_impl<ds_lie_type, ds_free_tensor_type>(*this, info));
//                case vector_type::dense:
//                    return free_tensor(sig_derivative_impl<dd_lie_type, dd_free_tensor_type>(*this, info));
//            }
//    }
//    throw std::invalid_argument("bad data");
}
free_tensor fallback_context::lie_to_tensor(const lie &arg) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(lie_to_tensor_impl<CTYPE, VTYPE>(*this, lie_base_access::get<lie_type<CTYPE, (VTYPE)>>(arg)), this)
ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
#undef ESIG_SWITCH_FN
//    switch (arg.coeff_type()) {
//        case coefficient_type::dp_real:
//            switch (arg.storage_type()) {
//                case vector_type::dense:
//                    return free_tensor(lie_to_tensor_impl<dd_lie_type, dd_free_tensor_type>(*this, lie_base_access::template get<dd_lie_type>(arg)));
//                case vector_type::sparse:
//                    return free_tensor(lie_to_tensor_impl<ds_lie_type, ds_free_tensor_type>(*this, lie_base_access::template get<ds_lie_type>(arg)));
//            }
//        case coefficient_type::sp_real:
//            switch (arg.storage_type()) {
//                case vector_type::dense:
//                    return free_tensor(lie_to_tensor_impl<sd_lie_type, sd_free_tensor_type>(*this, lie_base_access::template get<sd_lie_type>(arg)));
//                case vector_type::sparse:
//                    return free_tensor(lie_to_tensor_impl<ss_lie_type, ss_free_tensor_type>(*this, lie_base_access::template get<ss_lie_type>(arg)));
//            }
//    }
//    throw std::invalid_argument("bad data");
}
lie fallback_context::tensor_to_lie(const free_tensor &arg) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(tensor_to_lie_impl<CTYPE, VTYPE>(*this, free_tensor_base_access::get<ftensor_type<CTYPE, VTYPE>>(arg)), this)
ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
#undef ESIG_SWITCH_FN
//    switch(m_ctype) {
//        case coefficient_type::sp_real:
//            switch (arg.storage_type()) {
//                case vector_type::sparse:
//                    return lie(tensor_to_lie_impl<sd_lie_type>(*this, free_tensor_base_access::get<sd_free_tensor_type>(arg)));
//                case vector_type::dense:
//                    return lie(tensor_to_lie_impl<ss_lie_type>(*this, free_tensor_base_access::get<ss_free_tensor_type>(arg)));
//            }
//            break;
//        case coefficient_type::dp_real:
//            switch (arg.storage_type()) {
//                case vector_type::sparse:
//                    return lie(tensor_to_lie_impl<ds_lie_type>(*this, free_tensor_base_access::get<ds_free_tensor_type>(arg)));
//                case vector_type::dense:
//                    return lie(tensor_to_lie_impl<dd_lie_type>(*this, free_tensor_base_access::get<dd_free_tensor_type>(arg)));
//            }
//            break;
//    }
//    throw std::invalid_argument("bad data");
}

const std::vector<std::pair<key_type, int>>&
fallback_context::bracketing(const key_type &key) const
{
    assert(key < tensor_size(depth()));
    std::lock_guard<std::recursive_mutex> access(m_bracketing_lock);
    auto& found = m_bracketing_cache[key];
    if (found) {
        return static_cast<const std::vector<std::pair<key_type, int>>&>(found);
    }

    //found = std::make_unique<std::vector<std::pair<key_type, int>>>();
    //found.reset(new std::vector<std::pair<key_type, int>>());
    found.set();
    if (key == 0) {}
    else if (key <= m_width) {
        found.reserve(1);
        found.emplace_back(key, 1);
        found.shrink_to_fit();
    } else {

        auto kl = m_tensor_basis->lparent(key);
        auto kr = m_tensor_basis->rparent(key);
        for (const auto& outer : bracketing(kr)) {
            for (const auto& inner : m_lie_basis->prod(kl, outer.first)) {
                found.emplace_back(inner.first, outer.second*inner.second);
            }
        }
        found.shrink_to_fit();
    }
    return found;
}
const fallback_context::expand_result_t &fallback_context::expand(const key_type &key) const
{
    return m_expand(key);
}
coefficient_type fallback_context::ctype() const noexcept
{
    return m_ctype;
}
std::shared_ptr<context> fallback_context::get_alike(deg_t new_depth) const
{
    return fb_context_maker.maker->get_context(width(), new_depth, ctype());
}
std::shared_ptr<context> fallback_context::get_alike(coefficient_type new_coeff) const
{
    return fb_context_maker.maker->get_context(width(), depth(), new_coeff);
}
std::shared_ptr<context> fallback_context::get_alike(deg_t new_depth, coefficient_type new_coeff) const
{
    return fb_context_maker.maker->get_context(width(), new_depth, new_coeff);
}
std::shared_ptr<context> fallback_context::get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const
{
    return fb_context_maker.maker->get_context(new_width, new_depth, new_coeff);
}
free_tensor fallback_context::zero_tensor(vector_type vtype) const
{
    return context::zero_tensor(vtype);
}
lie fallback_context::zero_lie(vector_type vtype) const
{
    return context::zero_lie(vtype);
}
lie fallback_context::cbh(const std::vector<lie> &lies, vector_type vtype) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(cbh_impl<CTYPE, VTYPE>(*this, lies), this);
ESIG_MAKE_SWITCH(m_ctype, vtype)
#undef ESIG_SWITCH_FN
//    switch (m_ctype) {
//        case coefficient_type::sp_real:
//            switch (vtype) {
//                case vector_type::sparse:
//                    return lie(cbh_impl<sd_lie_type, sd_free_tensor_type>(*this, lies));
//                case vector_type::dense:
//                    return lie(cbh_impl<ss_lie_type, ss_free_tensor_type>(*this, lies));
//            }
//            break;
//        case coefficient_type::dp_real:
//            switch (vtype) {
//                case vector_type::sparse:
//                    return lie(cbh_impl<ds_lie_type, ds_free_tensor_type>(*this, lies));
//                case vector_type::dense:
//                    return lie(cbh_impl<dd_lie_type, dd_free_tensor_type>(*this, lies));
//            }
//            break;
//    }
//    throw std::invalid_argument("bad data");
}
std::shared_ptr<data_allocator> fallback_context::coefficient_alloc() const
{
    return m_coeff_alloc;
}
std::shared_ptr<data_allocator> fallback_context::pair_alloc() const
{
    return m_pair_alloc;
}
const algebra_basis &fallback_context::borrow_tbasis() const noexcept
{
    return *m_tensor_basis;
}
const algebra_basis &fallback_context::borrow_lbasis() const noexcept
{
    return *m_lie_basis;
}


std::shared_ptr<context> fallback_context_maker::get_context(deg_t width, deg_t depth, coefficient_type ctype) const
{
    return std::shared_ptr<context>(new fallback_context(width, depth, ctype));
}
bool fallback_context_maker::can_get(deg_t width, deg_t depth, coefficient_type ctype) const noexcept
{
    return true;
}
int fallback_context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept
{
    return 0;
}



} // namespace algebra
} // namespace esig
