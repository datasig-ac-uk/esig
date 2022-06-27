//
// Created by user on 18/03/2022.
//
#include <esig/algebra/free_tensor_interface.h>

namespace esig {
namespace algebra {

free_tensor::free_tensor(std::shared_ptr<free_tensor_interface> p)
    : p_impl(std::move(p))
{
}
const free_tensor_interface & free_tensor::operator*() const noexcept
{
    return *p_impl;
}
free_tensor_interface & free_tensor::operator*() noexcept
{
    return *p_impl;
}
dimn_t free_tensor::size() const
{
    return p_impl->size();
}
deg_t free_tensor::degree() const
{
    return p_impl->degree();
}
deg_t free_tensor::width() const
{
    return p_impl->width();
}
deg_t free_tensor::depth() const
{
    return p_impl->depth();
}
vector_type free_tensor::storage_type() const noexcept
{
    return p_impl->storage_type();
}
coefficient_type free_tensor::coeff_type() const noexcept
{
    return p_impl->coeff_type();
}
coefficient free_tensor::operator[](key_type key) const
{
    return p_impl->get(key);
}
coefficient free_tensor::operator[](key_type key)
{
    return p_impl->get_mut(key);
}
free_tensor free_tensor::uminus() const
{
    return p_impl->uminus();
}
free_tensor free_tensor::add(const free_tensor &other) const
{
    return p_impl->add(*other.p_impl);
}
free_tensor free_tensor::sub(const free_tensor &other) const
{
    return p_impl->sub(*other.p_impl);
}
free_tensor free_tensor::smul(const coefficient &other) const
{
    return p_impl->smul(other);
}
free_tensor free_tensor::sdiv(const coefficient &other) const
{
    return p_impl->sdiv(other);
}
free_tensor free_tensor::mul(const free_tensor &other) const
{
    return p_impl->mul(*other.p_impl);
}
free_tensor &free_tensor::add_inplace(const free_tensor &other)
{
    p_impl->add_inplace(*other.p_impl);
    return *this;
}
free_tensor &free_tensor::sub_inplace(const free_tensor &other)
{
    p_impl->sub_inplace(*other.p_impl);
    return *this;
}
free_tensor &free_tensor::smul_inplace(const coefficient &other)
{
    p_impl->smul_inplace(other);
    return *this;
}
free_tensor &free_tensor::sdiv_inplace(const coefficient &other)
{
    p_impl->sdiv_inplace(other);
    return *this;
}
free_tensor &free_tensor::mul_inplace(const free_tensor &other)
{
    p_impl->mul_inplace(*other.p_impl);
    return *this;
}
free_tensor &free_tensor::add_scal_mul(const free_tensor &other, const coefficient &scal)
{
    p_impl->add_scal_mul(*other.p_impl, scal);
    return *this;
}
free_tensor &free_tensor::sub_scal_mul(const free_tensor &other, const coefficient &scal)
{
    p_impl->sub_scal_mul(*other.p_impl, scal);
    return *this;
}
free_tensor &free_tensor::add_scal_div(const free_tensor &other, const coefficient &scal)
{
    p_impl->add_scal_div(*other.p_impl, scal);
    return *this;
}
free_tensor &free_tensor::sub_scal_div(const free_tensor &other, const coefficient &scal)
{
    p_impl->sub_scal_div(*other.p_impl, scal);
    return *this;
}
free_tensor &free_tensor::add_mul(const free_tensor &lhs, const free_tensor &rhs)
{
    p_impl->add_mul(*lhs.p_impl, *rhs.p_impl);
    return *this;
}
free_tensor &free_tensor::sub_mul(const free_tensor &lhs, const free_tensor & rhs)
{
    p_impl->sub_mul(*lhs.p_impl, *rhs.p_impl);
    return *this;
}
free_tensor &free_tensor::mul_smul(const free_tensor &other, const coefficient &scal)
{
    p_impl->mul_smul(*other.p_impl, scal);
    return *this;
}
free_tensor &free_tensor::mul_sdiv(const free_tensor &other, const coefficient &scal)
{
    p_impl->mul_sdiv(*other.p_impl, scal);
    return *this;
}
free_tensor free_tensor::exp() const
{
    return free_tensor(p_impl->exp());
}
free_tensor free_tensor::log() const
{
    return free_tensor(p_impl->log());
}
free_tensor free_tensor::inverse() const
{
    return free_tensor(p_impl->inverse());
}
free_tensor &free_tensor::fmexp(const free_tensor &other)
{
    p_impl->fmexp(*other.p_impl);
    return *this;
}
std::ostream &free_tensor::print(std::ostream &os) const
{
    p_impl->print(os);
    return os;
}
bool free_tensor::operator==(const free_tensor &other) const
{
    return p_impl->equals(*other.p_impl);
}
bool free_tensor::operator!=(const free_tensor &other) const
{
    return p_impl->equals(*other.p_impl);
}
std::ostream &operator<<(std::ostream &os, const free_tensor &arg)
{
    return arg.print(os);
}

algebra_iterator free_tensor::begin() const
{
    return p_impl->begin();
}
algebra_iterator free_tensor::end() const
{
    return p_impl->end();
}
dense_data_access_iterator free_tensor::iterate_dense_components() const noexcept {
    return p_impl->iterate_dense_components();
}

}

}
