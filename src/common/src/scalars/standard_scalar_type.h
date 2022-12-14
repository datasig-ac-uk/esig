//
// Created by user on 21/11/22.
//

#ifndef ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_
#define ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_

#include "esig/scalars.h"

#include <mutex>
#include <unordered_map>

namespace esig {
namespace scalars {

template<typename Scalar>
class standard_scalar_type : public scalar_type {

    mutable std::mutex m_lock;
    mutable std::unordered_map<std::string,
                               typename scalar_type::converter_function>
        m_converter_cache;

public:
    explicit standard_scalar_type(std::string id, std::string name)
        : scalar_type({std::move(id),
                       std::move(name),
                       sizeof(Scalar),
                       alignof(Scalar)}) {}

    void register_converter(const std::string &id, converter_function func) const noexcept override {
        std::lock_guard<std::mutex> access(m_lock);

        auto &cv = m_converter_cache[id];
        if (cv == nullptr) {
            cv = func;
        }
    }

    converter_function get_converter(const std::string &id) const noexcept override {
        std::lock_guard<std::mutex> access(m_lock);
        auto found = m_converter_cache.find(id);
        if (found != m_converter_cache.end()) {
            return found->second;
        }
        return nullptr;
    }

    scalar from(int i) const override {
        return scalar(Scalar(i), this);
    }
    scalar from(long long int numerator, long long int denominator) const override {
        return scalar(Scalar(numerator) / denominator, this);
    }

    scalar_pointer allocate(dimn_t size) const override {
        if (size == 1) {
            return scalar_pointer(new Scalar, this);
        } else {
            return scalar_pointer(new Scalar[size], this);
        }
    }
    void deallocate(scalar_pointer pointer, dimn_t size) const override {
        if (!pointer.is_null()) {
            if (size == 1) {
                delete pointer.template raw_cast<Scalar>();
            } else {
                delete[] pointer.template raw_cast<Scalar>();
            }
        }
    }

protected:
    Scalar try_convert(scalar_pointer other) const {
        if (other.is_null()) {
            return Scalar(0);
        }
        if (other.type() == this) {
            return *other.template raw_cast<const Scalar>();
        }

        const scalar_type *type = other.type();
        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        auto cv = this->get_converter(type->id());
        if (cv != nullptr) {
            Scalar result;
            cv(&result, other.ptr());
            return result;
        }

        throw std::runtime_error("could not convert " + type->info().name + " to scalar type " + info().name);
    }

public:
    void convert_copy(void *out, scalar_pointer in, dimn_t count) const override {
        assert(out != nullptr);
        assert(!in.is_null());
        const auto *type = in.type();

        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        if (type == this) {
            const auto *in_begin = in.template raw_cast<const Scalar>();
            const auto *in_end = in_begin + count;
            std::copy(in_begin, in_end, static_cast<Scalar *>(out));
        } else {
            auto cv = get_converter(type->id());
            auto *out_ptr = static_cast<Scalar *>(out);
            const auto *in_ptr = in.template raw_cast<const char>();
            const auto stride = in.type()->itemsize();

            while (count--) {
                cv(out_ptr++, in_ptr);
                in_ptr += stride;
            }
        }
    }

private:
    template<typename Basic>
    void convert_copy_basic(scalar_pointer &out,
                            const void *in,
                            dimn_t count) const noexcept {
        const auto *iptr = static_cast<const Basic *>(in);
        auto *optr = static_cast<Scalar *>(out.ptr());

        for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
            ::new (optr) Scalar(*iptr);
        }
    }

public:
    void convert_copy(scalar_pointer &out,
                      const void *in,
                      dimn_t count,
                      const std::string &type_id) const override {
        if (type_id == "f64") {
            return convert_copy_basic<double>(out, in, count);
        } else if (type_id == "f32") {
            return convert_copy_basic<float>(out, in, count);
        } else if (type_id == "i32") {
            return convert_copy_basic<int>(out, in, count);
        } else if (type_id == "u32") {
            return convert_copy_basic<unsigned int>(out, in, count);
        } else if (type_id == "i64") {
            return convert_copy_basic<long long>(out, in, count);
        } else if (type_id == "u64") {
            return convert_copy_basic<unsigned long long>(out, in, count);
        } else if (type_id == "isize") {
            return convert_copy_basic<std::ptrdiff_t>(out, in, count);
        } else if (type_id == "usize") {
            return convert_copy_basic<std::size_t>(out, in, count);
        } else if (type_id == "i16") {
            return convert_copy_basic<short>(out, in, count);
        } else if (type_id == "u16") {
            return convert_copy_basic<unsigned short>(out, in, count);
        } else if (type_id == "i8") {
            return convert_copy_basic<char>(out, in, count);
        } else if (type_id == "u8") {
            return convert_copy_basic<unsigned char>(out, in, count);
        }

        // If we're here, then it is a non-standard type
        auto otype = get_type(type_id);
        assert(otype != nullptr);
        convert_copy(out.ptr(), {in, otype}, count);
    }

    scalar convert(scalar_pointer other) const override {
        return scalar(try_convert(other), this);
    }
    scalar_t to_scalar_t(const void *arg) const override {
        return static_cast<scalar_t>(*static_cast<const Scalar *>(arg));
    }
    void assign(void *dst, scalar_pointer src) const override {
        auto *ptr = static_cast<Scalar *>(dst);
        if (src.is_null()) {
            *ptr = Scalar(0);
        } else {
            *ptr = try_convert(src);
        }
    }
    scalar copy(scalar_pointer arg) const override {
        return scalar(try_convert(arg), this);
    }
    scalar uminus(scalar_pointer arg) const override {
        return scalar(-try_convert(arg), this);
    }
    scalar add(const void *lhs, scalar_pointer rhs) const override {
        if (lhs == nullptr) {
            return copy(rhs);
        }
        return scalar(*static_cast<const Scalar *>(lhs) + try_convert(rhs), this);
    }
    scalar sub(const void *lhs, scalar_pointer rhs) const override {
        if (lhs == nullptr) {
            return uminus(rhs);
        }
        return scalar(*static_cast<const Scalar *>(lhs) - try_convert(rhs), this);
    }
    scalar mul(const void *lhs, scalar_pointer rhs) const override {
        if (lhs == nullptr) {
            return zero();
        }
        return scalar(*static_cast<const Scalar *>(lhs) * try_convert(rhs), this);
    }
    scalar div(const void *lhs, scalar_pointer rhs) const override {
        if (lhs == nullptr) {
            return zero();
        }
        if (rhs.is_null()) {
            throw std::runtime_error("division by zero");
        }
        return scalar(*static_cast<const Scalar *>(lhs) / try_convert(rhs), this);
    }
    bool are_equal(const void *lhs, const scalar_pointer &rhs) const noexcept override {
        return *static_cast<const Scalar *>(lhs) == try_convert(rhs);
    }

    scalar one() const override {
        return scalar(Scalar(1), this);
    }
    scalar mone() const override {
        return scalar(Scalar(-1), this);
    }
    scalar zero() const override {
        return scalar(Scalar(0), this);
    }
    void add_inplace(void *lhs, scalar_pointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<Scalar *>(lhs);
        *ptr += try_convert(rhs);
    }
    void sub_inplace(void *lhs, scalar_pointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<Scalar *>(lhs);
        *ptr -= try_convert(rhs);
    }
    void mul_inplace(void *lhs, scalar_pointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<Scalar *>(lhs);
        *ptr *= try_convert(rhs);
    }
    void div_inplace(void *lhs, scalar_pointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<Scalar *>(lhs);
        *ptr /= try_convert(rhs);
    }
    bool is_zero(const void *arg) const override {
        return arg == nullptr || *static_cast<const Scalar *>(arg) == Scalar(0);
    }
    void print(const void *arg, std::ostream &os) const override {
        if (arg == nullptr) {
            os << 0.0;
        } else {
            os << *static_cast<const Scalar *>(arg);
        }
    }
};

} // namespace scalars
}// namespace esig

#endif//ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_
