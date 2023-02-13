//
// Created by user on 21/03/2022.
//
#include <esig/algebra/lie_interface.h>


namespace esig {
namespace algebra {

//lie::lie(std::shared_ptr<lie_interface> p)
//    : p_impl(std::move(p))
//{
//}
//lie::lie(std::shared_ptr<lie_interface> &&p)
//    : p_impl(std::move(p))
//{
//}
//lie_interface &lie::operator*() noexcept
//{
//    return *p_impl;
//}
//const lie_interface &lie::operator*() const noexcept
//{
//    return *p_impl;
//}
//deg_t lie::width() const noexcept
//{
//    return p_impl->width();
//}
//
//deg_t lie::depth() const noexcept
//{
//    return p_impl->depth();
//}
//coefficient_type lie::coeff_type() const noexcept
//{
//    return p_impl->coeff_type();
//}
//vector_type lie::storage_type() const noexcept
//{
//    return p_impl->storage_type();
//}
//dimn_t lie::size() const noexcept
//{
//    return p_impl->size();
//}
//deg_t lie::degree() const noexcept
//{
//    return p_impl->degree();
//}
//
//algebra_iterator lie::begin() const
//{
//    return p_impl->begin();
//}
//algebra_iterator lie::end() const
//{
//    return p_impl->end();
//}
//
//lie lie::uminus() const
//{
//    return lie(p_impl->uminus());
//}
//lie lie::add(const lie &other) const
//{
//    return lie(p_impl->add(*other.p_impl));
//}
//lie lie::sub(const lie &other) const
//{
//    return lie(p_impl->sub(*other.p_impl));
//}
//lie lie::smul(const coefficient &other) const
//{
//    return lie(p_impl->smul(other));
//}
//lie lie::sdiv(const coefficient &other) const
//{
//    return lie(p_impl->sdiv(other));
//}
//lie lie::mul(const lie &other) const
//{
//    return lie(p_impl->mul(*other.p_impl));
//}
//lie &lie::add_inplace(const lie &other)
//{
//    p_impl->add_inplace(*other.p_impl);
//    return *this;
//}
//lie &lie::sub_inplace(const lie &other)
//{
//    p_impl->sub_inplace(*other.p_impl);
//    return *this;
//}
//lie &lie::smul_inplace(const coefficient &other)
//{
//    p_impl->smul_inplace(other);
//    return *this;
//}
//lie &lie::sdiv_inplace(const coefficient &other)
//{
//    p_impl->sdiv_inplace(other);
//    return *this;
//}
//lie &lie::mul_inplace(const lie &other)
//{
//    p_impl->mul_inplace(*other.p_impl);
//    return *this;
//}
//lie &lie::add_scal_mul(const lie &other, const coefficient &scal)
//{
//    p_impl->add_scal_mul(*other.p_impl, scal);
//    return *this;
//}
//lie &lie::sub_scal_mul(const lie &other, const coefficient &scal)
//{
//    p_impl->sub_scal_mul(*other.p_impl, scal);
//    return *this;
//}
//lie &lie::add_scal_div(const lie &other, const coefficient &scal)
//{
//    p_impl->add_scal_div(*other.p_impl, scal);
//    return *this;
//}
//lie &lie::sub_scal_div(const lie &other, const coefficient &scal)
//{
//    p_impl->sub_scal_div(*other.p_impl, scal);
//    return *this;
//}
//lie &lie::add_mul(const lie &lhs, const lie &rhs)
//{
//    p_impl->add_mul(*lhs.p_impl, *rhs.p_impl);
//    return *this;
//}
//lie &lie::sub_mul(const lie &lhs, const lie &rhs)
//{
//    p_impl->sub_mul(*lhs.p_impl, *rhs.p_impl);
//    return *this;
//}
//lie &lie::mul_smul(const lie &other, const coefficient &scal)
//{
//    p_impl->mul_smul(*other.p_impl, scal);
//    return *this;
//}
//lie &lie::mul_sdiv(const lie &other, const coefficient &scal)
//{
//    p_impl->mul_sdiv(*other.p_impl, scal);
//    return *this;
//}
//std::ostream &lie::print(std::ostream &os) const
//{
//    p_impl->print(os);
//    return os;
//}
//dense_data_access_iterator lie::iterate_dense_components() const {
//    return p_impl->iterate_dense_components();
//}
//
//coefficient lie::operator[](key_type key) const
//{
//    return p_impl->get(key);
//}
//coefficient lie::operator[](key_type key)
//{
//    return p_impl->get_mut(key);
//}
//
//
//bool lie::operator==(const lie &other) const
//{
//    return p_impl->equals(*other.p_impl);
//}
//bool lie::operator!=(const lie &other) const
//{
//    return !p_impl->equals(*other.p_impl);
//}
//
//std::ostream &operator<<(std::ostream &os, const lie &arg)
//{
//    return arg.print(os);
//}

} // namespace algebra
} // namespace esig
