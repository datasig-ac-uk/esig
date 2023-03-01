#ifndef ESIG_COMMON_SCALAR_INTERFACE_H_
#define ESIG_COMMON_SCALAR_INTERFACE_H_

#include "implementation_types.h"
#include "esig_export.h"

#include "scalar_fwd.h"

namespace esig { namespace scalars {

class ESIG_EXPORT ScalarInterface {
    friend class Scalar;

public:
    virtual ~ScalarInterface() = default;

    virtual const ScalarType *type() const noexcept = 0;

    virtual bool is_const() const noexcept = 0;
    virtual bool is_value() const noexcept = 0;
    virtual bool is_zero() const noexcept = 0;

    virtual scalar_t as_scalar() const = 0;
    virtual void assign(ScalarPointer) = 0;
    virtual void assign(const Scalar &other) = 0;
    virtual void assign(const void *data, const std::string &type_id) = 0;

    virtual ScalarPointer to_pointer() = 0;
    virtual ScalarPointer to_pointer() const noexcept = 0;
    virtual Scalar uminus() const;

    //    virtual scalar add(const scalar &other) const;
    //    virtual scalar sub(const scalar &other) const;
    //    virtual scalar mul(const scalar &other) const;
    //    virtual scalar div(const scalar &other) const;

    virtual void add_inplace(const Scalar &other) = 0;
    virtual void sub_inplace(const Scalar &other) = 0;
    virtual void mul_inplace(const Scalar &other) = 0;
    virtual void div_inplace(const Scalar &other) = 0;

    virtual bool equals(const Scalar &other) const noexcept;

    virtual std::ostream &print(std::ostream &os) const;
};
}}

#endif // ESIG_COMMON_SCALAR_INTERFACE_H_
