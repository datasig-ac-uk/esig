//
// Created by sam on 03/02/23.
//

#include "RandomScalarsFixture.h"


#include <vector>

namespace esig {
namespace testing {

scalars::OwnedScalarArray RandomScalars::random_data(const scalars::ScalarType *ctype, std::size_t count) {
    std::vector<float> tmp_data;

    tmp_data.reserve(count);
    for (std::size_t i=0; i<count; ++i) {
        tmp_data.push_back(dist(rng));
    }

    scalars::OwnedScalarArray result(ctype, count);
    scalars::ScalarPointer src(tmp_data.data());
    ctype->convert_copy(result.ptr(), src, count);

    return result;
}

}// namespace testing
}// namespace esig
