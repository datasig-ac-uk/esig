//
// Created by user on 03/11/22.
//

#include "esig/scalars.h"

using namespace esig;
using namespace esig::scalars;

scalar::scalar(scalar_pointer data, scalar::pointer_type ptype)
    : scalar_pointer(data), m_pointer_type(ptype) {
}
scalar::scalar(scalar_interface *other, interface_pointer)
    : scalar_pointer(other, other->type(),
                     other->is_const() ? IsConst : IsMutable),
      m_pointer_type(InterfacePointer) {
    if (p_data != nullptr) {
        p_type = other->type();
    } else {
        throw std::runtime_error("non-zero scalars must have a type");
    }
}
scalar::scalar(scalar_interface *other)
    : scalar_pointer(other, other->type(), other->is_const() ? IsConst : IsMutable),
      m_pointer_type(InterfacePointer)
{
}
scalar::scalar(scalar_pointer ptr)
    : scalar_pointer(ptr),
      m_pointer_type(BorrowedPointer) {
    if (p_data != nullptr && p_type == nullptr) {
        throw std::runtime_error("non-zero scalars must have a type");
    }
}

scalar::scalar(const scalar_type *type)
    : scalar_pointer(nullptr, type, IsMutable),
      m_pointer_type(OwnedPointer) {
}
scalar::scalar(scalar_t scal)
    : scalar_pointer(dtl::scalar_type_trait<scalar_t>::get_type()->allocate(1)),
      m_pointer_type(OwnedPointer) {
    p_type->assign(const_cast<void *>(p_data), {&scal, p_type});
}
scalar::scalar(scalar_t scal, const scalar_type *type)
    : scalar_pointer(type->allocate(1)),
      m_pointer_type(OwnedPointer) {
    const auto *scal_type = dtl::scalar_type_trait<scalar_t>::get_type();
    p_type->assign(const_cast<void *>(p_data), {&scal, scal_type});
}
scalar::scalar(const scalar &other)
    : scalar_pointer(other.p_type != nullptr ? scalar_pointer() : other.p_type->allocate(1)),
      m_pointer_type(other.m_pointer_type) {
    p_type->assign(const_cast<void *>(p_data), other.to_const_pointer());
}
scalar::scalar(scalar &&other) noexcept
    : scalar_pointer(std::move(other)),
      m_pointer_type(other.m_pointer_type) {
}

scalar::~scalar() {
    if (p_data != nullptr) {
        if (m_pointer_type == InterfacePointer) {
            delete static_cast<scalar_interface *>(const_cast<void *>(p_data));
        } else if (m_pointer_type == OwnedPointer) {
            p_type->deallocate({p_data, p_type}, 1);
        }
        p_data = nullptr;
    }
}

bool scalar::is_value() const noexcept {
    if (p_data == nullptr) {
        return true;
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const scalar_interface *>(p_data)->is_value();
    }

    return m_pointer_type == OwnedPointer;
}
bool scalar::is_zero() const noexcept {
    if (p_data == nullptr) {
        return true;
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const scalar_interface *>(p_data)->is_zero();
    }
    if (m_pointer_type == OwnedPointer) {
        return true;
    }

    // TODO: finish this off?
    return p_type->is_zero(p_data);
}

scalar &scalar::operator=(const scalar &other) {
    if (m_constness == IsConst) {
        throw std::runtime_error("Cannot cast to a const value");
    }
    if (this != std::addressof(other)) {
        if (m_pointer_type == InterfacePointer) {
            auto *iface = static_cast<scalar_interface *>(const_cast<void *>(p_data));
            iface->assign(other.to_const_pointer());
        } else {
            p_type->assign(const_cast<void *>(p_data), other.to_const_pointer());
        }
    }
    return *this;
}
scalar &scalar::operator=(scalar &&other) noexcept {
    if (this != std::addressof(other)) {
        if (p_type == nullptr || m_constness == IsConst) {
            this->~scalar();
            p_data = other.p_data;
            p_type = other.p_type;
            m_constness = other.m_constness;
            m_pointer_type = other.m_pointer_type;
            other.p_data = nullptr;
            other.p_type = nullptr;
            other.m_constness = IsConst;
            other.m_pointer_type = BorrowedPointer;
        } else {
            if (m_pointer_type == InterfacePointer) {
                auto* iface = static_cast<scalar_interface*>(const_cast<void*>(p_data));
                iface->assign(other.to_const_pointer());
            } else {
                p_type->assign(const_cast<void*>(p_data), other.to_const_pointer());
            }
        }
    }

    return *this;
}

scalar_pointer scalar::to_pointer() {
    if (m_constness == IsConst) {
        throw std::runtime_error("Cannot get mutable pointer to const object");
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<scalar_interface *>(
                   const_cast<void *>(p_data))
            ->to_pointer();
    }
    return {const_cast<void *>(p_data), p_type};
}
scalar_pointer scalar::to_const_pointer() const {
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const scalar_interface *>(p_data)->to_pointer();
    }
    return {p_data, p_type};
}
void scalar::set_to_zero() {
    if (p_data == nullptr) {
        assert(p_type != nullptr);
        assert(m_constness == IsMutable);
        assert(m_pointer_type == OwnedPointer);
        scalar_pointer::operator=(p_type->allocate(1));
        p_type->assign(const_cast<void *>(p_data), scalar_pointer());
    }
}
scalar_t scalar::to_scalar_t() const {
    if (p_data == nullptr) {
        return scalar_t(0);
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const scalar_interface *>(p_data)->as_scalar();
    }
    assert(p_type != nullptr);
    return p_type->to_scalar_t(p_data);
}
scalar scalar::operator-() const {
    if (p_data == nullptr) {
        return scalar(p_type);
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const scalar_interface *>(p_data)->uminus();
    }
    return p_type->uminus({p_data, p_type});
}

#define ESIG_SCALAR_OP(OP, MNAME)                                              \
    scalar scalar::operator OP(const scalar &other) const {                    \
        const scalar_type *type = (p_type != nullptr) ? p_type : other.p_type; \
        if (type == nullptr) {                                                 \
            return scalar();                                                   \
        }                                                                      \
        return type->MNAME(p_data, other.to_const_pointer());                  \
    }

ESIG_SCALAR_OP(+, add)
ESIG_SCALAR_OP(-, sub)
ESIG_SCALAR_OP(*, mul)
ESIG_SCALAR_OP(/, div)

#undef ESIG_SCALAR_OP

#define ESIG_SCALAR_IOP(OP, MNAME)                                                         \
    scalar &scalar::operator OP(const scalar &other) {                                     \
        if (m_constness == IsConst) {                                                      \
            throw std::runtime_error("performing inplace operation on const scalar");      \
        }                                                                                  \
                                                                                           \
        if (p_type == nullptr) {                                                           \
            assert(p_data == nullptr);                                                     \
            /* We just established that other.p_data != nullptr */                         \
            assert(other.p_type != nullptr);                                               \
            p_type = other.p_type;                                                         \
        }                                                                                  \
        if (p_data == nullptr) {                                                           \
            if (p_type == nullptr) {                                                       \
                p_type = other.p_type;                                                     \
            }                                                                              \
            set_to_zero();                                                                 \
        }                                                                                  \
        if (m_pointer_type == InterfacePointer) {                                          \
            auto *iface = static_cast<scalar_interface *>(const_cast<void *>(p_data));     \
            iface->MNAME##_inplace(other);                                                      \
        } else {                                                                           \
            p_type->MNAME##_inplace(const_cast<void *>(p_data), other.to_const_pointer()); \
        }                                                                                  \
        return *this;                                                                      \
    }

ESIG_SCALAR_IOP(+=, add)
ESIG_SCALAR_IOP(-=, sub)
ESIG_SCALAR_IOP(*=, mul)

scalar &scalar::operator/=(const scalar &other) {
    if (m_constness == IsConst) { throw std::runtime_error("performing inplace operation on const scalar"); }
    if (other.p_data == nullptr) {
        throw std::runtime_error("division by zero");
    }
    if (p_type == nullptr) {
        assert(p_data == nullptr);
        assert(other.p_type != nullptr);
        p_type = other.p_type;
    }
    if (p_data == nullptr) {
        if (p_type == nullptr) { p_type = other.p_type->rational_type(); }
        set_to_zero();
    }
    if (m_pointer_type == InterfacePointer) {
        auto *iface = static_cast<scalar_interface *>(const_cast<void *>(p_data));
        iface->div_inplace(other);
    } else {
        p_type->rational_type()->div_inplace(const_cast<void *>(p_data), other.to_const_pointer());
    }
    return *this;
}

#undef ESIG_SCALAR_IOP

bool scalar::operator==(const scalar &rhs) const noexcept {
    if (p_type == nullptr) {
        return rhs.is_zero();
    }
    return p_type->are_equal(p_data, rhs.to_const_pointer());
}
bool scalar::operator!=(const scalar &rhs) const noexcept {
    return !operator==(rhs);
}
std::ostream &esig::scalars::operator<<(std::ostream &os, const scalar &arg) {
    if (arg.p_type == nullptr) {
        os << '0';
    } else {
        arg.p_type->print(arg.p_data, os);
    }

    return os;
}
