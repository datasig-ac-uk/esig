//
// Created by user on 20/03/2022.
//

#include "fallback_context.h"
#include "lie_basis/lie_basis.h"
#include "tensor_basis/tensor_basis.h"
#include <esig/implementation_types.h>

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


} // namespace

std::shared_ptr<tensor_basis> esig::algebra::fallback_context::tbasis() const noexcept {
    return m_tensor_basis;
}
std::shared_ptr<lie_basis> esig::algebra::fallback_context::lbasis() const noexcept {
    return m_lie_basis;
}

static register_maker_helper<fallback_context_maker> fb_context_maker;



fallback_context::fallback_context(deg_t width, deg_t depth, coefficient_type ctype)
    : m_width(width), m_depth(depth), m_ctype(ctype),
      m_tensor_basis(new tensor_basis(width, depth)),
      m_lie_basis(new lie_basis(width, depth)),
      m_coeff_alloc(allocator_for_coeff(ctype)),
      m_pair_alloc(allocator_for_key_coeff(ctype)),
      m_maps(m_tensor_basis, m_lie_basis)
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
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(exp(log(compute_signature<CTYPE, VTYPE>(std::move(data)))), this)
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
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(tensor_to_lie_impl<CTYPE, VTYPE>(log(compute_signature<CTYPE, VTYPE>(std::move(data)))), this)
ESIG_MAKE_SWITCH(m_ctype, data.vtype())
#undef ESIG_SWITCH_FN
}
free_tensor fallback_context::sig_derivative(
        const std::vector<derivative_compute_info> &info,
        vector_type vtype,
        vector_type
        ) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(sig_derivative_impl<CTYPE, VTYPE>(info), this)
ESIG_MAKE_SWITCH(m_ctype, vtype)
#undef ESIG_SWITCH_FN
}
free_tensor fallback_context::lie_to_tensor(const lie &arg) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) free_tensor(lie_to_tensor_impl<CTYPE, VTYPE>(lie_base_access::get<lie_type<CTYPE, VTYPE>>(arg)), this)
ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
#undef ESIG_SWITCH_FN
}
lie fallback_context::tensor_to_lie(const free_tensor &arg) const
{
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(tensor_to_lie_impl<CTYPE, VTYPE>(free_tensor_base_access::get<ftensor_type<CTYPE, VTYPE>>(arg)), this)
ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
#undef ESIG_SWITCH_FN
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
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie(cbh_impl<CTYPE, VTYPE>(lies), this);
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
