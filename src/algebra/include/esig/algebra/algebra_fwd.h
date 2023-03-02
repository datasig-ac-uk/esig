#ifndef ESIG_ALGEBRA_ALGEBRA_FWD_H_
#define ESIG_ALGEBRA_ALGEBRA_FWD_H_

#include "esig/implementation_types.h"
#include "esig_algebra_export.h"

#include <boost/container/small_vector.hpp>

#include "esig/scalar_type.h"



namespace esig { namespace algebra {

enum class ImplementationType {
    borrowed,
    owned
};

enum class VectorType {
    sparse,
    dense
};

template<typename Interface>
struct algebra_access;

template<typename Basis>
struct basis_info {
    using this_key_type = typename Basis::key_type;

    static this_key_type convert_key(const Basis &basis, esig::key_type key);
    static esig::key_type convert_key(const Basis &basis, const this_key_type &key);

    static esig::key_type first_key(const Basis &basis);
    static esig::key_type last_key(const Basis &basis);

    static deg_t native_degree(const Basis &basis, const this_key_type &key);
    static deg_t degree(const Basis &basis, esig::key_type key);
};

template <typename Algebra>
struct algebra_info {
    using basis_type = typename Algebra::basis_type;
    using basis_traits = basis_info<basis_type>;
    using scalar_type = typename Algebra::scalar_type;
    using rational_type = scalar_type;
    using reference = scalar_type &;
    using const_reference = const scalar_type &;
    using pointer = scalar_type *;
    using const_pointer = const scalar_type *;

    static const scalars::ScalarType *ctype() noexcept { return ::esig::scalars::ScalarType::of<scalar_type>(); }
    static constexpr VectorType vtype() noexcept { return VectorType::sparse; }
    static deg_t width(const Algebra *instance) noexcept { return instance->basis().width(); }
    static deg_t max_depth(const Algebra *instance) noexcept { return instance->basis().depth(); }

    static const basis_type &basis(const Algebra &instance) noexcept { return instance.basis(); }

    using this_key_type = typename Algebra::key_type;
    static this_key_type convert_key(const Algebra *instance, esig::key_type key) noexcept { return basis_traits::convert_key(instance->basis(), key); }

    static deg_t degree(const Algebra &instance) noexcept { return instance.degree(); }
    static deg_t degree(const Algebra *instance, esig::key_type key) noexcept { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const Algebra *instance, this_key_type key) { return instance->basis().degree(key); }

    static key_type first_key(const Algebra *) noexcept { return 0; }
    static key_type last_key(const Algebra *instance) noexcept { return instance->basis().size(); }

    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container &key_product(const Algebra *inst, key_type k1, key_type k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container &key_product(const Algebra *inst, const this_key_type &k1, const this_key_type &k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }

    static Algebra create_like(const Algebra &instance) {
        return Algebra();
    }
};

}}
#endif // ESIG_ALGEBRA_ALGEBRA_FWD_H_
