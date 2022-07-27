//
// Created by user on 26/07/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
#include <esig/implementation_types.h>

namespace esig {
namespace algebra {

template<typename Ot, typename It>
void copy_convert(void *out, const void *in, dimn_t N) {
    auto* out_p = reinterpret_cast<Ot*>(out);
    const auto* in_p = reinterpret_cast<const It*>(in);
    for (dimn_t i = 0; i < N; ++i) {
        out_p[i] = key_type(in_p[i]);
    }
}

} // namespace algebra
} // namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
