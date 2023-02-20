//
// Created by user on 20/02/23.
//

#include "esig/scalars.h"

#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>

using namespace esig;
using namespace scalars;

static void f32_to_f64(void *dst, const void *src, dimn_t count);
static void f64_to_f32(void *dst, const void *src, dimn_t count);
static void i32_to_f32(void *dst, const void *src, dimn_t count);
static void i32_to_f64(void *dst, const void *src, dimn_t count);
static void i64_to_f32(void *dst, const void *src, dimn_t count);
static void i64_to_f64(void *dst, const void *src, dimn_t count);
static void i16_to_f32(void *dst, const void *src, dimn_t count);
static void i16_to_f64(void *dst, const void *src, dimn_t count);
static void i8_to_f32(void *dst, const void *src, dimn_t count);
static void i8_to_f64(void *dst, const void *src, dimn_t count);
static void isize_to_f32(void *dst, const void *src, dimn_t count);
static void isize_to_f64(void *dst, const void *src, dimn_t count);


static std::mutex conversion_lock;
using pair_type = std::pair<std::string, conversion_function>;

#define ADD_DEF_CONV(SRC, DST) pair_type { std::string("SRC->DST"), &SRC##_to_##DST }

static std::unordered_map<std::string, conversion_function> conversion_cache {
    ADD_DEF_CONV(f32, f64),
    ADD_DEF_CONV(f64, f32),
    ADD_DEF_CONV(i32, f32),
    ADD_DEF_CONV(i32, f64),
    ADD_DEF_CONV(i64, f32),
    ADD_DEF_CONV(i64, f64),
    ADD_DEF_CONV(i16, f32),
    ADD_DEF_CONV(i16, f64),
    ADD_DEF_CONV(i8 , f32),
    ADD_DEF_CONV(i8 , f64),
    ADD_DEF_CONV(isize, f32),
    ADD_DEF_CONV(isize, f64)
};

#undef ADD_DEF_CONV

static inline std::string type_ids_to_key(const std::string& src_type, const std::string& dst_type) {
    return src_type + "->" + dst_type;
}


conversion_function esig::scalars::get_conversion(const std::string& src_type, const std::string& dst_type) {
    std::lock_guard<std::mutex> access(conversion_lock);

    auto found = conversion_cache.find(type_ids_to_key(src_type, dst_type));
    if (found != conversion_cache.end()) {
        return found->second;
    }

    throw std::runtime_error("no conversion function from " + src_type + " to " + dst_type);
}

void register_conversion(const std::string &src_type, const std::string &dst_type, conversion_function func) {
    std::lock_guard<std::mutex> access(conversion_lock);

    auto& found = conversion_cache[type_ids_to_key(src_type, dst_type)];
    if (found != nullptr) {
        throw std::runtime_error("conversion from " + src_type + " to " + dst_type + " already registered");
    } else {
        found = func;
    }
}

static void f32_to_f64(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<double *>(dst);
    const auto *in = static_cast<const float *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) double(in[i]);
    }
}

static void f64_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const double *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(static_cast<float>(in[i]));
    }
}

void i32_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const int *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(static_cast<float>(in[i]));
    }
}
void i32_to_f64(void *dst, const void *src, dimn_t count) {
    auto* out = static_cast<double*>(dst);
    const auto* in = static_cast<const int*>(src);

    for (dimn_t i=0; i<count; ++i) {
        :: new (out++) double(in[i]);
    }
}
void i64_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const long long *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(in[i]);
    }
}
void i64_to_f64(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<double *>(dst);
    const auto *in = static_cast<const long long *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) double(in[i]);
    }
}
void i16_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const short *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(in[i]);
    }
}
void i16_to_f64(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<double *>(dst);
    const auto *in = static_cast<const short *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) double(in[i]);
    }
}
void i8_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const char *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(in[i]);
    }
}
void i8_to_f64(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<double *>(dst);
    const auto *in = static_cast<const char*>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) double(in[i]);
    }
}
void isize_to_f32(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<float *>(dst);
    const auto *in = static_cast<const std::ptrdiff_t *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) float(in[i]);
    }
}
void isize_to_f64(void *dst, const void *src, dimn_t count) {
    auto *out = static_cast<double *>(dst);
    const auto *in = static_cast<const std::ptrdiff_t *>(src);

    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) double(in[i]);
    }
}
