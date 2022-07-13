//
// Created by user on 05/07/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_DETAIL_CONVERTING_LIE_ITERATOR_ADAPTOR_H_
#define ESIG_SRC_ALGEBRA_SRC_DETAIL_CONVERTING_LIE_ITERATOR_ADAPTOR_H_

#include <esig/algebra/lie_interface.h>

#include <vector>


namespace esig {
namespace algebra {
namespace dtl {

template <typename OutLie>
class converting_reference_wrapper
{
    const lie_interface* m_interface;
    OutLie tmp;
    bool converted;

    void convert()
    {
        using scal_t = typename OutLie::scalar_type;
        for (auto &item : *m_interface) {
            tmp.add_scal_prod(item.key(), coefficient_cast<scal_t>(item.value()));
        }
        converted = true;
    }

public:

    explicit converting_reference_wrapper(
        std::shared_ptr<lie_basis> basis,
        const lie_interface* interface)
        : m_interface(interface), tmp(std::move(basis))
    {}

    operator const OutLie&()
    {
        if (typeid(m_interface) == typeid(OutLie)) {
            return dynamic_cast<const OutLie&>(*m_interface);
        } else {
            if (!converted) {
                convert();
            }
            return tmp;
        }
    }

    operator const OutLie*()
    {
        if (typeid(m_interface) == typeid(OutLie)) {
            return static_cast<const OutLie*>(m_interface);
        } else {
            if (!converted) {
                convert();
            }
            return &tmp;
        }
    }
};


template <typename OutLie>
class converting_lie_iterator_adaptor
{
    using base_iterator = std::vector<lie>::const_iterator;
    base_iterator m_data;
    std::shared_ptr<lie_basis> m_basis;
public:
    using difference_type = std::ptrdiff_t;
    using value_type = OutLie;
    using reference = converting_reference_wrapper<OutLie>;
    using pointer = converting_reference_wrapper<OutLie>;
    using iterator_category = std::forward_iterator_tag;


    explicit converting_lie_iterator_adaptor(
        std::shared_ptr<lie_basis> basis,
        base_iterator it) : m_data(it), m_basis(std::move(basis))
    {}

    converting_lie_iterator_adaptor& operator++()
    {
        ++m_data;
        return *this;
    }

    const converting_lie_iterator_adaptor operator++(int)
    {
        return converting_lie_iterator_adaptor(m_data++);
    }

    bool operator==(const converting_lie_iterator_adaptor& other) const
    {
        return m_data == other.m_data;
    }

    bool operator!=(const converting_lie_iterator_adaptor& other) const
    {
        return m_data != other.m_data;
    }

    reference operator*()
    {
        return reference(m_basis, algebra_base_access::get(*m_data));
    }
};

}// namespace dtl
}// namespace algebra
}// namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_DETAIL_CONVERTING_LIE_ITERATOR_ADAPTOR_H_
