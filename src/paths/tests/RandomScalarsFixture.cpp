//
// Created by sam on 03/02/23.
//

#include "RandomScalarsFixture.h"


#include <vector>

namespace esig {
namespace testing {

scalars::owned_scalar_array RandomScalars::random_data(const scalars::scalar_type *ctype, std::size_t count) {
    std::vector<float> tmp_data;

    tmp_data.reserve(count);
    for (std::size_t i=0; i<count; ++i) {
        tmp_data.push_back(dist(rng));
    }

    scalars::owned_scalar_array result(ctype, count);
    scalars::scalar_pointer src(tmp_data.data());
    ctype->convert_copy(result.ptr(), src, count);

    return result;
}

}// namespace testing
}// namespace esig
