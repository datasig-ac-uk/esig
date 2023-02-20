//
// Created by user on 10/02/23.
//

#ifndef ESIG_SRC_PYESIG_PY_CONTEXT_H_
#define ESIG_SRC_PYESIG_PY_CONTEXT_H_

#include <memory>
#include <pybind11/pybind11.h>
#include <esig/algebra/context.h>

namespace esig { namespace python {

class py_context {
    std::shared_ptr<const algebra::context> p_ctx;

public:
    py_context(std::shared_ptr<const algebra::context> ctx) : p_ctx(ctx) {}

    operator const std::shared_ptr<const algebra::context> &() const noexcept { return p_ctx; }

    const algebra::context &operator*() const noexcept { return *p_ctx; }
    const algebra::context *operator->() const noexcept { return p_ctx.get(); }

    std::shared_ptr<const algebra::context> get_pointer() const { return p_ctx; }

};


void init_context(pybind11::module_& m);

}}
#endif//ESIG_SRC_PYESIG_PY_CONTEXT_H_
