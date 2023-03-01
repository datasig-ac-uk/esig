//
// Created by user on 20/02/23.
//

#include "esig/scalars.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

using namespace esig;
using namespace scalars;

#define MAKE_CONVERSION_FUNCTION(SRC, DST, SRC_T, DST_T)                             \
    static void SRC##_to_##DST(ScalarPointer dst, ScalarPointer src, dimn_t count) { \
        const auto *src_p = src.raw_cast<const SRC_T>();                             \
        auto *dst_p = dst.raw_mut_cast<DST_T>();                                     \
                                                                                     \
        for (dimn_t i = 0; i < count; ++i) {                                         \
            ::new (dst_p++) DST_T(src_p[i]);                                         \
        }                                                                            \
    }

MAKE_CONVERSION_FUNCTION(f32, f64, float, double)
MAKE_CONVERSION_FUNCTION(f64, f32, double, float)
MAKE_CONVERSION_FUNCTION(i32, f32, int, float)
MAKE_CONVERSION_FUNCTION(i32, f64, int, double)
MAKE_CONVERSION_FUNCTION(i64, f32, long long, float)
MAKE_CONVERSION_FUNCTION(i64, f64, long long, double)
MAKE_CONVERSION_FUNCTION(i16, f32, short, float)
MAKE_CONVERSION_FUNCTION(i16, f64, short, double)
MAKE_CONVERSION_FUNCTION(i8, f32, char, float)
MAKE_CONVERSION_FUNCTION(i8, f64, char, double)
MAKE_CONVERSION_FUNCTION(isize, f32, idimn_t, float)
MAKE_CONVERSION_FUNCTION(isize, f64, idimn_t, double)

#undef MAKE_CONVERSION_FUNCTION

static std::mutex conversion_lock;
using pair_type = std::pair<std::string, conversion_function>;

#define ADD_DEF_CONV(SRC, DST) \
    pair_type { std::string(#SRC "->" #DST), conversion_function(&SRC##_to_##DST) }

static std::unordered_map<std::string, conversion_function> conversion_cache{
    ADD_DEF_CONV(f32, f64),
    ADD_DEF_CONV(f64, f32),
    ADD_DEF_CONV(i32, f32),
    ADD_DEF_CONV(i32, f64),
    ADD_DEF_CONV(i64, f32),
    ADD_DEF_CONV(i64, f64),
    ADD_DEF_CONV(i16, f32),
    ADD_DEF_CONV(i16, f64),
    ADD_DEF_CONV(i8, f32),
    ADD_DEF_CONV(i8, f64),
    ADD_DEF_CONV(isize, f32),
    ADD_DEF_CONV(isize, f64)};

#undef ADD_DEF_CONV

static inline std::string type_ids_to_key(const std::string &src_type, const std::string &dst_type) {
    return src_type + "->" + dst_type;
}

const conversion_function &esig::scalars::get_conversion(const std::string &src_type, const std::string &dst_type) {
    std::lock_guard<std::mutex> access(conversion_lock);

    auto found = conversion_cache.find(type_ids_to_key(src_type, dst_type));
    if (found != conversion_cache.end()) {
        return found->second;
    }

    throw std::runtime_error("no conversion function from " + src_type + " to " + dst_type);
}

void register_conversion(const std::string &src_type, const std::string &dst_type, conversion_function func) {
    std::lock_guard<std::mutex> access(conversion_lock);

    auto &found = conversion_cache[type_ids_to_key(src_type, dst_type)];
    if (found != nullptr) {
        throw std::runtime_error("conversion from " + src_type + " to " + dst_type + " already registered");
    } else {
        found = std::move(func);
    }
}
