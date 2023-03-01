#ifndef ESIG_COMMON_SCALAR_ARRAY_H_
#define ESIG_COMMON_SCALAR_ARRAY_H_

#include "implementation_types.h"
#include "esig_export.h"

#include "scalar_fwd.h"
#include "scalar_pointer.h"
#include "scalar.h"

namespace esig { namespace scalars {


class ScalarArray : public ScalarPointer {

protected:
    dimn_t m_size;

public:
    ScalarArray() : ScalarPointer(), m_size(0) {}

    ScalarArray(const ScalarArray &other) noexcept;
    ScalarArray(ScalarArray &&other) noexcept;

    explicit ScalarArray(const ScalarType *type);
    ScalarArray(void *data, const ScalarType *type, dimn_t size);
    ScalarArray(const void *data, const ScalarType *type, dimn_t size);
    ScalarArray(ScalarPointer begin, dimn_t size)
        : ScalarPointer(begin), m_size(size) {}

    ScalarArray &operator=(const ScalarArray &other) noexcept;
    ScalarArray &operator=(ScalarArray &&other) noexcept;

    template<typename Int>
    Scalar operator[](Int index) {
        auto uindex = static_cast<dimn_t>(index);
        assert(0 <= uindex && uindex < m_size);
        return ScalarPointer::operator[](uindex);
    }

    template<typename Int>
    Scalar operator[](Int index) const noexcept {
        auto uindex = static_cast<dimn_t>(index);
        assert(0 <= uindex && uindex < m_size);
        return ScalarPointer::operator[](uindex);
    }

    constexpr dimn_t size() const noexcept { return m_size; }
};

}}

#endif // ESIG_COMMON_SCALAR_ARRAY_H_
