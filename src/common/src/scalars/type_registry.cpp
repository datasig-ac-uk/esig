//
// Created by user on 15/11/22.
//


#include "esig/scalars.h"

#include <unordered_map>
#include <mutex>

using namespace esig;
using namespace esig::scalars;

/*
 * All types should support conversion from the following,
 * but these are not represented as scalar types by themselves.
 * The last 3 are commented out, because they are to be implemented
 * separately.
 */
static const std::string reserved[] = {
    "i32",                // int
    "u32",                // unsigned int
    "i64",                // long long
    "u64",                // unsigned long long
//    "l",                // long
//    "L",                // unsigned long
    "isize",              // ssize_t
    "usize",              // size_t
    "i16",                // short
    "u16",                // unsigned short
    "i8",                 // char
    "u8",                 // unsigned char
//    "c",                // char
//    "e",                // float16
//    "g",                // float128
//    "O"                 // Object
};
static std::mutex scalar_type_cache_lock;
static std::unordered_map<std::string, const ScalarType *> scalar_type_cache;


void esig::scalars::register_type(const std::string& identifier, const ScalarType * type)
{
    std::lock_guard<std::mutex> access(scalar_type_cache_lock);

    for (const auto & i : reserved) {
        if (identifier == i) {
            throw std::runtime_error("cannot register identifier " + identifier);
        }
    }

    auto& entry = scalar_type_cache[identifier];
    if (entry != nullptr) {
        throw std::runtime_error("cannot register type with identifier " + identifier);
    }
    entry = type;
}

const ScalarType * esig::scalars::get_type(const std::string& identifier)
{
    std::lock_guard<std::mutex> access(scalar_type_cache_lock);
    auto found = scalar_type_cache.find(identifier);
    if (found != scalar_type_cache.end()) {
        return found->second;
    }
    throw std::runtime_error("no registered_type with identifier " + identifier);
}

namespace {

struct register_helper
{
    register_helper() {
        register_type(std::string("f32"), dtl::scalar_type_holder<float>::get_type());
        register_type(std::string("f64"), dtl::scalar_type_holder<double>::get_type());
        register_type(std::string("rational"), dtl::scalar_type_holder<rational_scalar_type>::get_type());
    }
};

} // namespace

static const register_helper basic_types_registration;
