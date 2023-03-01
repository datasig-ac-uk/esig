#ifndef ESIG_COMMON_SCALAR_TYPE_H_
#define ESIG_COMMON_SCALAR_TYPE_H_

#include "esig_export.h"
#include "implementation_types.h"

#include <functional>
#include <string>

#include "scalar_fwd.h"
#include "scalar_pointer.h"

namespace esig {
namespace scalars {

using conversion_function = std::function<void(ScalarPointer, ScalarPointer, dimn_t)>;

class ESIG_EXPORT ScalarType {
    const ScalarTypeInfo m_info;

public:
    using converter_function = void (*)(void *, const void *);

    explicit ScalarType(ScalarTypeInfo info)
        : m_info(std::move(info)) {}

    // No copy/move constructors or assigments
    ScalarType(const ScalarType &) = delete;
    ScalarType(ScalarType &&) noexcept = delete;
    ScalarType &operator=(const ScalarType &) = delete;
    ScalarType &operator=(ScalarType &&) noexcept = delete;

    virtual ~ScalarType();

    template<typename T>
    static const ScalarType *of() noexcept {
        return dtl::scalar_type_holder<T>::get_type();
    }

    static const ScalarType *from_type_details(const BasicScalarTypeDetails &details);
    static const ScalarType *for_id(const std::string &id);

    virtual Scalar from(int) const = 0;
    virtual Scalar from(long long, long long) const = 0;

    const std::string &id() const noexcept { return m_info.id; }
    const ScalarTypeInfo &info() const noexcept { return m_info; }
    int alignment() const noexcept { return m_info.alignment; }
    int itemsize() const noexcept { return m_info.size; }

    virtual const ScalarType *rational_type() const noexcept;

    virtual ScalarPointer allocate(dimn_t) const = 0;
    virtual void deallocate(ScalarPointer, dimn_t) const = 0;
    virtual void convert_copy(void *out, ScalarPointer in, dimn_t count) const = 0;
    virtual void convert_copy(ScalarPointer &out, const void *in, dimn_t count, const std::string &type_id) const = 0;
    virtual void convert_copy(ScalarPointer out, const void *in, dimn_t count, const BasicScalarTypeDetails &details) const;
    virtual Scalar convert(ScalarPointer other) const = 0;

    virtual Scalar one() const;
    virtual Scalar mone() const;
    virtual Scalar zero() const;

    virtual scalar_t to_scalar_t(const void *arg) const = 0;

    virtual void assign(void *dst, ScalarPointer src) const = 0;
    virtual void assign(void *dst, ScalarPointer src, dimn_t count) const;
    virtual void assign(void *dst, long long numerator, long long denominator) const;
    virtual Scalar copy(ScalarPointer arg) const = 0;
    virtual Scalar uminus(ScalarPointer arg) const = 0;
    virtual Scalar add(const void *lhs, ScalarPointer rhs) const = 0;
    virtual Scalar sub(const void *lhs, ScalarPointer rhs) const = 0;
    virtual Scalar mul(const void *lhs, ScalarPointer rhs) const = 0;
    virtual Scalar div(const void *lhs, ScalarPointer rhs) const = 0;

    virtual void add_inplace(void *lhs, ScalarPointer rhs) const;
    virtual void sub_inplace(void *lhs, ScalarPointer rhs) const;
    virtual void mul_inplace(void *lhs, ScalarPointer rhs) const;
    virtual void div_inplace(void *lhs, ScalarPointer rhs) const;

    virtual bool is_zero(const void *) const;
    virtual bool are_equal(const void *, const ScalarPointer &rhs) const noexcept = 0;

    virtual void print(const void *, std::ostream &os) const;
};

ESIG_EXPORT void register_type(const std::string &identifier, const ScalarType *type);
ESIG_EXPORT const ScalarType *get_type(const std::string &identifier);

ESIG_EXPORT
const conversion_function& get_conversion(const std::string &src_type, const std::string &dst_type);
ESIG_EXPORT
void register_conversion(const std::string &src_type, const std::string &dst_type, conversion_function func);

ESIG_EXPORT
inline bool operator==(const ScalarType &lhs, const ScalarType &rhs) noexcept { return &lhs == &rhs; }

ESIG_EXPORT
inline bool operator!=(const ScalarType &lhs, const ScalarType &rhs) noexcept { return &lhs == &rhs; }
}// namespace scalars
}// namespace esig

#endif// ESIG_COMMON_SCALAR_TYPE_H_
