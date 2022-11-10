//
// Created by sam on 10/11/22.
//


#include "esig/scalars.h"


using namespace esig;


scalar scalar_pointer::operator*() noexcept
{
    assert(p_data != nullptr && p_type != nullptr);
    return p_type->dereference_pointer(p_data);
}

scalar scalar_pointer::operator*() const noexcept {
    assert(p_data != nullptr && p_type != nullptr);
    return p_type->dereference_pointer(p_data);
}
scalar_pointer scalar_pointer::operator+(dimn_t index) const noexcept
{
    auto* ptr = static_cast<char*>(p_data);
    auto itemsize = p_type->m_info.size;
    ptr += index*itemsize;
    return { ptr, p_type };
}
