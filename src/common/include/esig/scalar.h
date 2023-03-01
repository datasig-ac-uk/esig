#ifndef ESIG_COMMON_SCALAR_H_
#define ESIG_COMMON_SCALAR_H_

#include "implementation_types.h"
#include "esig_export.h"

#include <iosfwd>
#include <type_traits>
#include <utility>

#include "scalar_type.h"
#include "scalar_pointer.h"
#include "scalar_interface.h"

namespace esig { namespace scalars {

ESIG_EXPORT
std::ostream &operator<<(std::ostream &os, const Scalar &arg);

class ESIG_EXPORT Scalar : private ScalarPointer {

    template<typename T>
    friend class dtl::scalar_type_trait;

    //    const void* p_impl;
    //    const scalar_type* p_type;

    //    enum constness {
    //        IsConst,
    //        IsMutable
    //    } m_constness = IsConst;

    using ScalarPointer::IsConst;
    using ScalarPointer::IsMutable;
    using ScalarPointer::m_constness;

    struct InterfacePointerMarker {};

    enum pointer_type {
        OwnedPointer,   // A raw pointer to a scalar, onwed by this
        BorrowedPointer,// A raw pointer to a scalar, borrowed from elsewhere
        InterfacePointer// A pointer to a scalar_interface type
    } m_pointer_type = OwnedPointer;

    explicit Scalar(ScalarInterface *other, InterfacePointerMarker);

    Scalar(ScalarPointer data, pointer_type ptype);

public:
    Scalar() = default;
    explicit Scalar(const ScalarType *);
    explicit Scalar(scalar_t);
    Scalar(scalar_t, const ScalarType *);
    explicit Scalar(ScalarPointer ptr);
    explicit Scalar(ScalarInterface *other);

    template<typename I,
             typename J,
             typename = std::enable_if_t<
                 std::is_integral<I>::value && std::is_integral<J>::value>>
    Scalar(I numerator, J denominator, const ScalarType *type)
        : ScalarPointer((type == nullptr ? get_type("rational") : type)->allocate(1)) {
        p_type->assign(const_cast<void *>(p_data),
                       static_cast<long long>(numerator),
                       static_cast<long long>(denominator));
    }

    template<typename ScalarImpl>
    Scalar(ScalarImpl arg, const ScalarType *type)
        : ScalarPointer((type == nullptr ? dtl::scalar_type_trait<ScalarImpl>::get_type() : type)->allocate(1)),
          m_pointer_type(OwnedPointer) {
        const auto *arg_type = dtl::scalar_type_trait<ScalarImpl>::get_type();
        p_type->assign(const_cast<void *>(p_data), {std::addressof(arg), arg_type});
    }

    Scalar(const Scalar &other);
    Scalar(Scalar &&other) noexcept;

    ~Scalar();

    Scalar &operator=(const Scalar &other);
    Scalar &operator=(Scalar &&other) noexcept;

    template<typename T, typename = std::enable_if_t<!std::is_same<std::decay_t<T>, Scalar>::value>>
    Scalar &operator=(T arg) {
        if (m_constness == IsConst) {
            throw std::runtime_error("attempting to assign to const value");
        }

        assert(p_type != nullptr);
        if (p_data == nullptr) {
            m_pointer_type = OwnedPointer;
            p_data = p_type->allocate(1).ptr();
        }

        const auto &type_id = type_id_of<T>();
        if (m_pointer_type == InterfacePointer) {
            static_cast<ScalarInterface *>(const_cast<void *>(p_data))
                ->assign(std::addressof(arg), type_id);
        } else {
            auto ptr = to_pointer();
            p_type->convert_copy(ptr, std::addressof(arg), 1, type_id);
        }

        return *this;
    }

    using ScalarPointer::is_const;
    //    bool is_const() const noexcept;
    bool is_value() const noexcept;
    bool is_zero() const noexcept;

    using ScalarPointer::type;
    //    const scalar_type *type() const noexcept;

    ScalarPointer to_pointer();
    ScalarPointer to_const_pointer() const;
    void set_to_zero();

    scalar_t to_scalar_t() const;

    Scalar operator-() const;

    Scalar operator+(const Scalar &other) const;
    Scalar operator-(const Scalar &other) const;
    Scalar operator*(const Scalar &other) const;
    Scalar operator/(const Scalar &other) const;

    Scalar &operator+=(const Scalar &other);
    Scalar &operator-=(const Scalar &other);
    Scalar &operator*=(const Scalar &other);
    Scalar &operator/=(const Scalar &other);

    bool operator==(const Scalar &rhs) const noexcept;
    bool operator!=(const Scalar &rhs) const noexcept;

    friend std::ostream &operator<<(std::ostream &os, const Scalar &arg);
};


}}


#endif // ESIG_COMMON_SCALAR_H_
