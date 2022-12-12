//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_INTERVAL_H_
#define ESIG_SRC_PYESIG_INTERVAL_H_

#include "py_esig.h"
#include <esig/intervals.h>

namespace esig { namespace python {

struct py_interval : public esig::interval {
    using interval::interval;

    param_t inf() const override;
    param_t sup() const override;
    param_t included_end() const override;
    param_t excluded_end() const override;

    bool contains(param_t arg) const noexcept override;
    bool is_associated(const interval &arg) const noexcept override;
    bool contains(const interval &arg) const noexcept override;
};


void init_py_interval(py::module_& m);

}
}

#endif//ESIG_SRC_PYESIG_INTERVAL_H_
