#ifndef ESIG_ALGEBRA_FREE_TENSOR_H_
#define ESIG_ALGEBRA_FREE_TENSOR_H_

#include "algebra_fwd.h"

#include "algebra_base.h"

namespace esig {
namespace algebra {

class FreeTensor;

extern template class ESIG_ALGEBRA_EXPORT AlgebraInterface<FreeTensor>;

class ESIG_ALGEBRA_EXPORT FreeTensorInterface
    : public AlgebraInterface<FreeTensor> {
public:
    using algebra_interface_t = AlgebraInterface<FreeTensor>;


    // Special functions
    virtual FreeTensor exp() const = 0;
    virtual FreeTensor log() const = 0;
    virtual FreeTensor inverse() const = 0;
    virtual FreeTensor antipode() const = 0;
    virtual void fmexp(const FreeTensor &other) = 0;
};

template <typename, template <typename> class>
class FreeTensorImplementation;

extern template class ESIG_ALGEBRA_EXPORT AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;


class ESIG_ALGEBRA_EXPORT FreeTensor : public AlgebraBase<FreeTensorInterface, FreeTensorImplementation> {

    using base_t = AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

public:
    using base_t::base_t;

    FreeTensor exp() const { return p_impl->exp(); }
    FreeTensor log() const { return p_impl->log(); }
    FreeTensor inverse() const { return p_impl->inverse(); }
    FreeTensor antipode() const { return p_impl->antipode(); }
    FreeTensor& fmexp(const FreeTensor& other) { p_impl->fmexp(other); return *this; }

};


}// namespace algebra
}// namespace esig

#endif// ESIG_ALGEBRA_FREE_TENSOR_H_
