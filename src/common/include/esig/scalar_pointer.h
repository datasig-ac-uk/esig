//
// Created by user on 27/02/23.
//

#ifndef ESIG_SRC_COMMON_INCLUDE_ESIG_SCALAR_POINTER_H_
#define ESIG_SRC_COMMON_INCLUDE_ESIG_SCALAR_POINTER_H_

#include "implementation_types.h"
#include "esig_export.h"

#include "scalar_fwd.h"

namespace esig {
namespace scalars {


class ESIG_EXPORT ScalarPointer {

public:
    enum Constness {
        IsConst,
        IsMutable
    };

protected:
    const void *p_data;
    const ScalarType *p_type;
    Constness m_constness = IsConst;

    ScalarPointer(const void *, const ScalarType *, Constness);

public:
    using difference_type = std::ptrdiff_t;

    const ScalarType *type() const noexcept { return p_type; }

    ScalarPointer() : p_data(nullptr), p_type(nullptr), m_constness(IsMutable) {}
    explicit ScalarPointer(const ScalarType *type) : p_data(nullptr), p_type(type) {}

    ScalarPointer(void *ptr, const ScalarType *type)
        : p_data(ptr), p_type(type), m_constness(IsMutable) {}
    ScalarPointer(const void *ptr, const ScalarType *type)
        : p_data(ptr), p_type(type), m_constness(IsConst) {}

    template<typename T>
    explicit ScalarPointer(T *ptr)
        : p_data(ptr), p_type(dtl::scalar_type_trait<T>::get_type()), m_constness(IsMutable) {}
    template<typename T>
    explicit ScalarPointer(const T *ptr)
        : p_data(ptr), p_type(dtl::scalar_type_trait<T>::get_type()), m_constness(IsConst) {}

    const void *ptr() const noexcept { return p_data; }
    void *ptr() noexcept { return const_cast<void *>(p_data); }

    template<typename T>
    const T *raw_cast() const noexcept {
        return static_cast<const T *>(p_data);
    }

    template <typename T>
    T* raw_mut_cast() const {
        if (m_constness == IsConst) {
            throw std::runtime_error("cannot conversion const pointer to non-const pointer");
        }
        return static_cast<T*>(const_cast<void*>(p_data));
    }

    bool is_null() const noexcept { return p_data == nullptr; }
    bool is_const() const noexcept { return m_constness == IsConst; }

    Scalar deref();
    Scalar deref_mut();

    Scalar operator*();
    Scalar operator*() const noexcept;

    ScalarPointer operator+(dimn_t index) const noexcept;
    ScalarPointer &operator+=(dimn_t index) noexcept;

    difference_type operator-(const ScalarPointer &other) const noexcept;

    ScalarPointer &operator++() noexcept;
    const ScalarPointer operator++(int) noexcept;

    Scalar operator[](dimn_t index) const noexcept;
    Scalar operator[](dimn_t index);

    bool operator==(const ScalarPointer &other) const noexcept { return p_type == other.p_type && p_data == other.p_data; }
    bool operator!=(const ScalarPointer &other) const noexcept { return p_type != other.p_type || p_data != other.p_data; }
};



}// namespace scalars
}// namespace esig

#endif//ESIG_SRC_COMMON_INCLUDE_ESIG_SCALAR_POINTER_H_
