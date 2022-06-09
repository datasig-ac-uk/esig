//
// Created by user on 21/03/2022.
//
#include <esig/algebra/iteration.h>



namespace esig {
namespace algebra {


algebra_iterator::algebra_iterator()
    : p_impl(nullptr), p_ctx(nullptr)
{
}
algebra_iterator::algebra_iterator(const algebra_iterator &arg)
    : p_impl(arg.p_impl), p_ctx(arg.p_ctx)
{
}
algebra_iterator::algebra_iterator(algebra_iterator &&arg) noexcept
    : p_impl(std::move(arg.p_impl)), p_ctx(std::move(arg.p_ctx))
{
}

algebra_iterator &algebra_iterator::operator++()
{
    p_impl->advance();
    return *this;
}
const algebra_iterator_item &algebra_iterator::operator*() const
{
    return p_impl->get();
}
algebra_iterator::pointer algebra_iterator::operator->() const
{
    return p_impl->get_ptr();
}
bool algebra_iterator::operator==(const algebra_iterator &other) const
{
    return p_impl->equals(*other.p_impl);
}
bool algebra_iterator::operator!=(const algebra_iterator &other) const
{
    return !operator==(other);
}

const context* algebra_iterator::get_context() const noexcept
{
    return p_ctx;
}


} // namespace algebra
} // namespace esig
