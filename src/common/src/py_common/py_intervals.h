//
// Created by user on 04/05/22.
//

#ifndef ESIG_SRC_COMMON_SRC_PY_COMMON_PY_INTERVALS_H_
#define ESIG_SRC_COMMON_SRC_PY_COMMON_PY_INTERVALS_H_

#include <esig/intervals.h>

namespace esig {


struct py_interval : public esig::interval
{
    using interval::interval;

    param_t inf() const override;
    param_t sup() const override;
    param_t included_end() const override;
    param_t excluded_end() const override;

    bool contains(param_t arg) const noexcept override;
    bool is_associated(const interval &arg) const noexcept override;
    bool contains(const interval &arg) const noexcept override;
};


} // namespace esig



#endif//ESIG_SRC_COMMON_SRC_PY_COMMON_PY_INTERVALS_H_
