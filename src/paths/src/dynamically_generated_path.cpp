//
// Created by user on 05/04/2022.
//

#include "dynamically_generated_path.h"

namespace esig {
namespace paths {





bool dynamically_constructed_path::empty(const interval &domain) const
{
    return false;
}
algebra::lie dynamically_constructed_path::log_signature(const interval &domain, const algebra::context &ctx) const
{
//    std::cout << domain << '\n';
//    const auto& md = metadata();
//    const auto buf = eval(domain);
//
//    const auto* ptr = reinterpret_cast<const double*>(buf.begin());
//    const auto* end = reinterpret_cast<const double*>(buf.end());
////    for (; ptr != end; ++ptr) {
////        std::cout << *ptr << ' ';
////    }
////    std::cout << '\n';
//
//    esig::algebra::signature_data data(
//            md.ctype,
//            md.result_vec_type,
//            dtl::function_increment_iterator(buf.begin(), buf.end())
//            );

//    auto result = ctx.log_signature(std::move(data));
//    std::cout << result << '\n';
    return ctx.zero_lie(algebra::vector_type::dense);
}


} // namespace paths
} // namespace esig
