#ifndef ESIG_COMMON_KEY_SCALAR_ARRAY_H_
#define ESIG_COMMON_KEY_SCALAR_ARRAY_H_

#include "esig_export.h"
#include "implementation_types.h"

#include "scalar_fwd.h"
#include "scalar_array.h"

namespace esig {
namespace scalars {

class ESIG_EXPORT KeyScalarArray : public ScalarArray {

    const key_type *p_keys = nullptr;
    bool m_scalars_owned = false;
    bool m_keys_owned = true;

public:
    KeyScalarArray() = default;
    ~KeyScalarArray();

    KeyScalarArray(const KeyScalarArray &other);
    KeyScalarArray(KeyScalarArray &&other) noexcept;

    explicit KeyScalarArray(OwnedScalarArray &&sa) noexcept;
    KeyScalarArray(ScalarArray base, const key_type *keys);

    explicit KeyScalarArray(const ScalarType *type) noexcept;
    explicit KeyScalarArray(const ScalarType *type, dimn_t n) noexcept;
    KeyScalarArray(const ScalarType *type, const void *begin, dimn_t count) noexcept;

    explicit operator OwnedScalarArray() &&noexcept;

    KeyScalarArray &operator=(const ScalarArray &other) noexcept;
    KeyScalarArray &operator=(KeyScalarArray &&other) noexcept;
    KeyScalarArray &operator=(OwnedScalarArray &&other) noexcept;

    const key_type *keys() const noexcept { return p_keys; }
    key_type *keys();
    bool has_keys() const noexcept { return p_keys != nullptr; }

    void allocate_scalars(idimn_t count = -1);
    void allocate_keys(idimn_t count = -1);
};

}// namespace scalars
}// namespace esig

#endif// ESIG_COMMON_KEY_SCALAR_ARRAY_H_
