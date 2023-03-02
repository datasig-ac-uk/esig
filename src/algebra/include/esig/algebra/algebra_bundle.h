#ifndef ESIG_ALGEBRA_ALGEBRA_BUNDLE_H_
#define ESIG_ALGEBRA_ALGEBRA_BUNDLE_H_

#include "algebra_fwd.h"

#include "algebra_base.h"

namespace esig { namespace algebra{

template<typename Base, typename Fibre>
struct AlgebraBundleInterface : public Base {
    using base_t = Base;
    using fibre_t = Fibre;
    using base_interface_t = typename Base::interface_t;
    using fibre_interface_t = typename Fibre::interface_t;

    virtual Fibre fibre() = 0;
};


//template <typename BundleInterface, template <typename> class AlgebraWrapper=AlgebraBase>
//class ESIG_ALGEBRA_EXPORT AlgebraBundle : public AlgebraWrapper<BundleInterface> {
//    static_assert(std::is_base_of<AlgebraBase<BundleInterface>, AlgebraWrapper<BundleInterface>>::value,
//                  "wrapping class must be a derivative of AlgebraBase"
//                  );
//public:
//
//    using base_t = typename BundleInterface::base_t;
//    using fibre_t = typename BundleInterface::fibre_t;
//    using base_interface_t = typename BundleInterface::base_interface_t;
//    using fibre_interface_t = typename BundleInterface::fibre_interface_t;
//
//
//
//
//    fibre_t fibre();
//
//
//};


}}


#endif // ESIG_ALGEBRA_ALGEBRA_BUNDLE_H_
