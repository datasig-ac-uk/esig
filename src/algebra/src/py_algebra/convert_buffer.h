//
// Created by user on 26/07/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
#include <esig/implementation_types.h>

namespace esig {
namespace algebra {

template<typename Ot, typename It>
void copy_convert(char *out, const void *in, dimn_t N) noexcept {
    auto* out_p = reinterpret_cast<Ot*>(out);
    const auto* in_p = reinterpret_cast<const It*>(in);
    for (dimn_t i = 0; i < N; ++i) {
        out_p[i] = key_type(in_p[i]);
    }
}

template <typename Ot, typename KIt, typename VIt>
void copy_kv_convert(char* out, const void* kin, const void* vin, dimn_t N) noexcept
{
    auto* out_p = reinterpret_cast<Ot*>(out);
    const auto* kin_p = reinterpret_cast<const KIt*>(kin);
    const auto* vin_p = reinterpret_cast<const VIt*>(vin);
    for (dimn_t i=0; i<N; ++i) {
        out_p[i] = Ot(kin_p[i], vin_p[i]);
    }
}


template <typename T>
std::shared_ptr<data_allocator> allocator_for()
{
    return std::shared_ptr<data_allocator>(new dtl::allocator_ext<std::allocator<T>>());
}

} // namespace algebra
} // namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_CONVERT_BUFFER_H_
