//
// Created by user on 05/04/2022.
//

#include "libalgebra_context_maker.h"
#include "register_la_contexts.h"
#include <esig/libalgebra_context/libalgebra_context.h>

#include <functional>
#include <limits>

namespace {

using esig::deg_t;

template <deg_t Width, deg_t Depth, esig::algebra::coefficient_type CType>
std::shared_ptr<const esig::algebra::context> build_ctx()
{
    using ctx_t = esig::algebra::libalgebra_context<Width, Depth, CType>;
    return std::shared_ptr<const esig::algebra::context>(new ctx_t());
}

//std::shared_ptr<esig::algebra::context> get_ctx(deg_t width, deg_t depth)
//{
//#define TemplatedFn(width, depth) build_ctx<width, depth>()
//#include "switch.inl"
//#undef TemplatedFn
//}




}

const esig::algebra::libalgebra_context_maker::context_map &esig::algebra::libalgebra_context_maker::get() const
{
    return cache;
}

esig::algebra::libalgebra_context_maker::libalgebra_context_maker()
{
    esig::algebra::dtl::register_la_contexts(cache);
}

bool esig::algebra::libalgebra_context_maker::can_get(esig::deg_t width, esig::deg_t depth, coefficient_type ctype) const noexcept
{
    std::lock_guard<std::recursive_mutex> access(m_lock);
    auto found = cache.find({width, depth, ctype});
    return found != cache.end();
}
int esig::algebra::libalgebra_context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept
{
    return 1;
}
std::shared_ptr<const esig::algebra::context> esig::algebra::libalgebra_context_maker::get_context(esig::deg_t width, esig::deg_t depth, esig::algebra::coefficient_type ctype) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);
    auto found = cache.find({width, depth, ctype});
    if (found != cache.end()) {
        return found->second;
    }
    throw std::runtime_error("no context found");
}
