//
// Created by user on 21/11/22.
//

#ifndef ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_
#define ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_

#include "esig/scalars.h"


namespace esig {
namespace scalars {

template<typename ScalarImpl>
class StandardScalarType : public ScalarType {
public:
    explicit StandardScalarType(std::string id, std::string name)
        : ScalarType({std::move(id),
                       std::move(name),
                       sizeof(ScalarImpl),
                       alignof(ScalarImpl)}) {}



    Scalar from(int i) const override {
        return Scalar(ScalarImpl(i), this);
    }
    Scalar from(long long int numerator, long long int denominator) const override {
        return Scalar(ScalarImpl(numerator) / denominator, this);
    }

    ScalarPointer allocate(dimn_t size) const override {
        if (size == 1) {
            return ScalarPointer(new ScalarImpl, this);
        } else {
            return ScalarPointer(new ScalarImpl[size], this);
        }
    }
    void deallocate(ScalarPointer pointer, dimn_t size) const override {
        if (!pointer.is_null()) {
            if (size == 1) {
                delete pointer.template raw_cast<ScalarImpl>();
            } else {
                delete[] pointer.template raw_cast<ScalarImpl>();
            }
        }
    }

protected:
    ScalarImpl try_convert(ScalarPointer other) const {
        if (other.is_null()) {
            return ScalarImpl(0);
        }
        if (other.type() == this) {
            return *other.template raw_cast<const ScalarImpl>();
        }

        const ScalarType *type = other.type();
        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        auto cv = get_conversion(type->id(), this->id());
        if (cv) {
            ScalarImpl result;
            ScalarPointer result_ptr {&result, this};
            cv(result_ptr, other, 1);
            return result;
        }

        throw std::runtime_error("could not convert " + type->info().name + " to scalar type " + info().name);
    }

public:
    void convert_copy(void *out, ScalarPointer in, dimn_t count) const override {
        assert(out != nullptr);
        assert(!in.is_null());
        const auto *type = in.type();

        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        if (type == this) {
            const auto *in_begin = in.template raw_cast<const ScalarImpl>();
            const auto *in_end = in_begin + count;
            std::copy(in_begin, in_end, static_cast<ScalarImpl *>(out));
        } else {
            auto cv = get_conversion(type->id(), this->id());
            ScalarPointer out_ptr {out, this};

            cv(out_ptr, in, count);
        }
    }

private:
    template<typename Basic>
    void convert_copy_basic(ScalarPointer &out,
                            const void *in,
                            dimn_t count) const noexcept {
        const auto *iptr = static_cast<const Basic *>(in);
        auto *optr = static_cast<ScalarImpl *>(out.ptr());

        for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
            ::new (optr) ScalarImpl(*iptr);
        }
    }

public:
    void convert_copy(ScalarPointer &out,
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

    Scalar convert(ScalarPointer other) const override {
        return Scalar(try_convert(other), this);
    }
    scalar_t to_scalar_t(const void *arg) const override {
        return static_cast<scalar_t>(*static_cast<const ScalarImpl *>(arg));
    }
    void assign(void *dst, ScalarPointer src) const override {
        auto *ptr = static_cast<ScalarImpl *>(dst);
        if (src.is_null()) {
            *ptr = ScalarImpl(0);
        } else {
            *ptr = try_convert(src);
        }
    }
    Scalar copy(ScalarPointer arg) const override {
        return Scalar(try_convert(arg), this);
    }
    Scalar uminus(ScalarPointer arg) const override {
        return Scalar(-try_convert(arg), this);
    }
    Scalar add(const void *lhs, ScalarPointer rhs) const override {
        if (lhs == nullptr) {
            return copy(rhs);
        }
        return Scalar(*static_cast<const ScalarImpl *>(lhs) + try_convert(rhs), this);
    }
    Scalar sub(const void *lhs, ScalarPointer rhs) const override {
        if (lhs == nullptr) {
            return uminus(rhs);
        }
        return Scalar(*static_cast<const ScalarImpl *>(lhs) - try_convert(rhs), this);
    }
    Scalar mul(const void *lhs, ScalarPointer rhs) const override {
        if (lhs == nullptr) {
            return zero();
        }
        return Scalar(*static_cast<const ScalarImpl *>(lhs) * try_convert(rhs), this);
    }
    Scalar div(const void *lhs, ScalarPointer rhs) const override {
        if (lhs == nullptr) {
            return zero();
        }
        if (rhs.is_null()) {
            throw std::runtime_error("division by zero");
        }
        return Scalar(*static_cast<const ScalarImpl *>(lhs) / try_convert(rhs), this);
    }
    bool are_equal(const void *lhs, const ScalarPointer &rhs) const noexcept override {
        return *static_cast<const ScalarImpl *>(lhs) == try_convert(rhs);
    }

    Scalar one() const override {
        return Scalar(ScalarImpl(1), this);
    }
    Scalar mone() const override {
        return Scalar(ScalarImpl(-1), this);
    }
    Scalar zero() const override {
        return Scalar(ScalarImpl(0), this);
    }
    void add_inplace(void *lhs, ScalarPointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<ScalarImpl *>(lhs);
        *ptr += try_convert(rhs);
    }
    void sub_inplace(void *lhs, ScalarPointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<ScalarImpl *>(lhs);
        *ptr -= try_convert(rhs);
    }
    void mul_inplace(void *lhs, ScalarPointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<ScalarImpl *>(lhs);
        *ptr *= try_convert(rhs);
    }
    void div_inplace(void *lhs, ScalarPointer rhs) const override {
        assert(lhs != nullptr);
        auto *ptr = static_cast<ScalarImpl *>(lhs);
        *ptr /= try_convert(rhs);
    }
    bool is_zero(const void *arg) const override {
        return arg == nullptr || *static_cast<const ScalarImpl *>(arg) == ScalarImpl(0);
    }
    void print(const void *arg, std::ostream &os) const override {
        if (arg == nullptr) {
            os << 0.0;
        } else {
            os << *static_cast<const ScalarImpl *>(arg);
        }
    }
};

} // namespace scalars
}// namespace esig

#endif//ESIG_SRC_COMMON_SRC_SCALARS_STANDARD_SCALAR_TYPE_H_
