#ifndef ESIG_ALGEBRA_SHUFFLE_TENSOR_H_
#define ESIG_ALGEBRA_SHUFFLE_TENSOR_H_

#include "algebra_fwd.h"
#include "algebra_base.h"

namespace esig { namespace algebra {

class ShuffleTensor;

extern template class ESIG_ALGEBRA_EXPORT AlgebraInterface<ShuffleTensor>;

using ShuffleTensorInterface = AlgebraInterface<ShuffleTensor>;

extern template class ESIG_ALGEBRA_EXPORT AlgebraBase<ShuffleTensorInterface>;

class ESIG_ALGEBRA_EXPORT ShuffleTensor : public AlgebraBase<ShuffleTensorInterface> {};



}}

#endif // ESIG_ALGEBRA_SHUFFLE_TENSOR_H_
