#ifndef ESIG_ALGEBRA_FREE_TENSOR_IMPL_H_
#define ESIG_ALGEBRA_FREE_TENSOR_IMPL_H_

#include "algebra_fwd.h"
#include "algebra_impl.h"
#include "free_tensor.h"

namespace esig {
namespace algebra {

namespace dtl {

template<typename Tensor>
Tensor exp_wrapper(const Tensor &arg) {
    return exp(arg);
}

template<typename Tensor>
Tensor log_wrapper(const Tensor &arg) {
    return log(arg);
}

template<typename Tensor>
Tensor inverse_wrapper(const Tensor &arg) {
    return inverse(arg);
}

template <typename Tensor>
Tensor antipode_wrapper(const Tensor& arg) {
    return antipode(arg);
}

}// namespace dtl

template<typename Impl, template <typename> class StorageModel>
class FreeTensorImplementation : public AlgebraImplementation<FreeTensorInterface, Impl, StorageModel> {

    using base_t = AlgebraImplementation<FreeTensorInterface, Impl, StorageModel>;

    friend class algebra_access<FreeTensorInterface>;
    friend class algebra_access<AlgebraInterface<FreeTensor>>;
protected:
    using base_t::data;
    using base_t::cast;
public:
    using base_t::base_t;

    FreeTensor exp() const override {
        return FreeTensor(dtl::exp_wrapper(FreeTensorImplementation::data()), FreeTensorImplementation::p_ctx);
    }
    FreeTensor log() const override {
        return FreeTensor(dtl::log_wrapper(FreeTensorImplementation::data()), FreeTensorImplementation::p_ctx);
    }
    FreeTensor inverse() const override {
        return FreeTensor(dtl::inverse_wrapper(FreeTensorImplementation::data()), FreeTensorImplementation::p_ctx);
    }
    FreeTensor antipode() const override {
        return FreeTensor(dtl::antipode_wrapper(FreeTensorImplementation::data()), FreeTensorImplementation::p_ctx);
    }


    void fmexp(const FreeTensor& other) override {
        auto& self = base_t::data();
        self.fmexp_inplace(base_t::template cast<const Impl&>(*other));
    }
};





}// namespace algebra
}// namespace esig

#endif// ESIG_ALGEBRA_FREE_TENSOR_IMPL_H_
