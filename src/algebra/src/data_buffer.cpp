//
// Created by sam on 09/05/22.
//

#include <esig/algebra/coefficients.h>

namespace esig {
namespace algebra {


data_buffer::data_buffer(char *begin, char *end, bool constant)
    : data_begin(begin), data_end(end), is_const(constant)
{
}
data_buffer::data_buffer(const char *begin, const char *end)
    : data_begin(const_cast<char*>(begin)), data_end(const_cast<char*>(end)), is_const(true)
{
}
data_buffer::pointer data_buffer::begin() noexcept
{
    if (is_const) {
        return nullptr;
    } else {
        return reinterpret_cast<char *>(data_begin);
    }
}
data_buffer::pointer data_buffer::end() noexcept
{
    if (is_const) {
        return nullptr;
    } else {
        return reinterpret_cast<char*>(data_end);
    }
}
data_buffer::const_pointer data_buffer::begin() const noexcept
{
    return reinterpret_cast<const char*>(data_begin);
}
data_buffer::const_pointer data_buffer::end() const noexcept
{
    return reinterpret_cast<const char*>(data_end);
}
data_buffer::size_type data_buffer::size() const noexcept
{
    return static_cast<size_type>(data_end - data_begin);
}
void data_buffer::swap(data_buffer &other) noexcept {
    std::swap(data_begin, other.data_begin);
    std::swap(data_end, other.data_end);
    std::swap(is_const, other.is_const);
}

data_buffer &data_buffer::operator=(const data_buffer &other) noexcept {
    data_begin = other.data_begin;
    data_end = other.data_end;
    is_const = other.is_const;
    return *this;
}
data_buffer &data_buffer::operator=(data_buffer &&other) noexcept {
    data_begin = other.data_begin;
    data_end = other.data_end;
    is_const = other.is_const;

    // set to nullptr to avoid problems
    other.data_begin = nullptr;
    other.data_end = nullptr;
    return *this;
}

}// namespace algebra
}// namespace esig
