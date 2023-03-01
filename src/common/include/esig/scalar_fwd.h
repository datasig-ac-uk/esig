#ifndef ESIG_COMMON_SCALAR_FWD_H_
#define ESIG_COMMON_SCALAR_FWD_H_

#include "implementation_types.h"

#include <functional>
#include <string>


#include "config.h"

namespace esig {
namespace scalars {

struct DeviceInfo {
    std::int32_t type;
    std::int32_t device_id;
};

struct BasicScalarTypeDetails {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
    DeviceInfo device;
};

struct ScalarTypeInfo {
    std::string id;
    std::string name;
    int size;
    int alignment;

    BasicScalarTypeDetails info;
};

class ScalarType;
class Scalar;
class ScalarPointer;
class ScalarArray;
class ScalarStream;
class OwnedScalarArray;
class KeyScalarArray;



namespace dtl {

template<typename T>
class scalar_type_trait;

template<typename T>
struct type_id_of_impl;

template<typename ScalarImpl>
struct scalar_type_holder {
    static const ScalarType *get_type() noexcept;
};

template <typename T>
class ScalarImplementation;

}// namespace dtl


struct unsigned_size_type_marker {};
struct signed_size_type_marker {};


template<typename T>
constexpr const std::string &type_id_of() noexcept {
    return dtl::type_id_of_impl<T>::get_id();
}

}// namespace scalars
}// namespace esig

#endif// ESIG_COMMON_SCALAR_FWD_H_
