#ifndef ESIG_ALGEBRA_LIE_H_
#define ESIG_ALGEBRA_LIE_H_

#include "algebra_fwd.h"
#include "algebra_base.h"


namespace esig { namespace algebra {

class Lie;
extern template class ESIG_ALGEBRA_EXPORT AlgebraInterface<Lie>;

using LieInterface = AlgebraInterface<Lie>;

extern template class ESIG_ALGEBRA_EXPORT AlgebraBase<LieInterface>;

class ESIG_ALGEBRA_EXPORT Lie : public AlgebraBase<LieInterface>
{
    using base_t = AlgebraBase<LieInterface>;

public:

    using base_t::base_t;
};

}}

#endif // ESIG_ALGEBRA_LIE_H_
