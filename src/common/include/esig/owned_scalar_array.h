#ifndef ESIG_COMMON_OWNED_SCALAR_ARRAY_H_
#define ESIG_COMMON_OWNED_SCALAR_ARRAY_H_

#include "implementation_types.h"
#include "esig_export.h"


#include "scalar_fwd.h"
#include "scalar_array.h"

namespace esig { namespace scalars {


class ESIG_EXPORT OwnedScalarArray : public ScalarArray {
public:
    OwnedScalarArray() = default;

    OwnedScalarArray(const OwnedScalarArray &other);
    OwnedScalarArray(OwnedScalarArray &&other) noexcept;

    explicit OwnedScalarArray(const ScalarType *type);
    OwnedScalarArray(const ScalarType *type, dimn_t size);
    OwnedScalarArray(const Scalar &value, dimn_t count);
    explicit OwnedScalarArray(const ScalarArray &other);

    explicit OwnedScalarArray(const ScalarType *type, const void *data, dimn_t count) noexcept;

    OwnedScalarArray &operator=(const ScalarArray &other);
    OwnedScalarArray &operator=(OwnedScalarArray &&other) noexcept;

    ~OwnedScalarArray();
};

}}


#endif // ESIG_COMMON_OWNED_SCALAR_ARRAY_H_
