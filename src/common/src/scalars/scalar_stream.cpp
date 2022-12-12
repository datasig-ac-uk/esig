//
// Created by user on 16/11/22.
//
#include "esig/scalars.h"

using namespace esig;
using namespace esig::scalars;

scalar_stream::scalar_stream() : m_stream(), m_elts_per_row(0), p_type(nullptr)
{
}

scalar_stream::scalar_stream(const scalar_type *type)
    : m_stream(), m_elts_per_row(0), p_type(type)
{
}
scalar_stream::scalar_stream(scalar_pointer base, std::vector<dimn_t> shape)
{
    if (!base.is_null()) {
        p_type = base.type();
        if (p_type == nullptr) {
            throw std::runtime_error("missing type");
        }
        if (shape.empty()) {
            throw std::runtime_error("strides cannot be empty");
        }

        const auto* ptr = static_cast<const char*>(base.ptr());
        const auto itemsize = p_type->itemsize();

        dimn_t rows = shape[0];
        dimn_t cols = (shape.size() > 1) ? shape[1] : 1;

        m_elts_per_row.push_back(cols);

        dimn_t stride = cols*itemsize;
        m_stream.reserve(rows);

        for (dimn_t i=0; i<rows; ++i) {
            m_stream.push_back(ptr);
            ptr += stride;
        }
    }
}

dimn_t scalar_stream::col_count(esig::dimn_t i) const noexcept
{
    if (m_elts_per_row.size() == 1) {
        return m_elts_per_row[0];
    }

    assert(m_elts_per_row.size() > 1);
    assert(i < m_elts_per_row.size());
    return m_elts_per_row[i];
}

scalar_array scalar_stream::operator[](dimn_t row) const noexcept {
    return { scalar_pointer(m_stream[row], p_type), col_count(row) };
}
scalar scalar_stream::operator[](std::pair<dimn_t, dimn_t> index) const noexcept {
    auto first = operator[](index.first);
    return first[index.second];
}
