//
// Created by sam on 07/03/2022.
//

#ifndef ESIG_PATHS_BASIS_H_
#define ESIG_PATHS_BASIS_H_

#include "algebra_fwd.h"


#include <boost/optional.hpp>

namespace esig {
namespace algebra {


class ESIG_ALGEBRA_EXPORT BasisInterface {
public:
    virtual ~BasisInterface() = default;

    virtual boost::optional<deg_t> width() const noexcept;
    virtual boost::optional<deg_t> depth() const noexcept;
    virtual boost::optional<deg_t> degree(key_type key) const noexcept;
    virtual std::string key_to_string(key_type key) const noexcept;
    virtual dimn_t size(int) const noexcept;
    virtual dimn_t start_of_degree(int) const noexcept;
    virtual boost::optional<key_type> lparent(key_type key) const noexcept;
    virtual boost::optional<key_type> rparent(key_type key) const noexcept;
    virtual key_type index_to_key(dimn_t idx) const noexcept;
    virtual dimn_t key_to_index(key_type key) const noexcept;
    virtual let_t first_letter(key_type key) const noexcept;
    virtual key_type key_of_letter(let_t letter) const noexcept;
    virtual bool letter(key_type key) const noexcept;
};

template<typename Impl>
class BasisImplementation : public BasisInterface {
    const Impl *p_basis;
    using traits = basis_info<Impl>;

public:
    BasisImplementation(const Impl *arg) : p_basis(arg) {}

    boost::optional<deg_t> width() const noexcept override {
        return p_basis->width();
    }
    boost::optional<deg_t> depth() const noexcept override {
        return p_basis->depth();
    }
    boost::optional<deg_t> degree(key_type key) const noexcept override {
        return traits::degree(*p_basis, key);
    }
    std::string key_to_string(key_type key) const noexcept override {
        std::stringstream ss;
        p_basis->print_key(ss, traits::convert_key(*p_basis, key));
        return ss.str();
    }
    dimn_t size(int i) const noexcept override {
        return p_basis->size(i);
    }
    dimn_t start_of_degree(int i) const noexcept override {
        return p_basis->start_of_degree(i);
    }
    boost::optional<key_type> lparent(key_type key) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->lparent(traits::convert_key(*p_basis, key)));
    }
    boost::optional<key_type> rparent(key_type key) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->rparent(traits::convert_key(*p_basis, key)));
    }
    key_type index_to_key(dimn_t idx) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->index_to_key(idx));
    }
    dimn_t key_to_index(key_type key) const noexcept override {
        return dimn_t(p_basis->key_to_index(traits::convert_key(*p_basis, key)));
    }
    let_t first_letter(key_type key) const noexcept override {
        return p_basis->first_letter(traits::convert_key(*p_basis, key));
    }
    key_type key_of_letter(let_t letter) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->key_of_letter(letter));
    }
    bool letter(key_type key) const noexcept override {
        return p_basis->letter(traits::convert_key(*p_basis, key));
    }
};

class ESIG_ALGEBRA_EXPORT Basis {
    std::unique_ptr<const BasisInterface> p_impl;

public:
    template<typename T>
    explicit Basis(const T *arg) : p_impl(new BasisImplementation<T>(arg)) {}

    const BasisInterface &operator*() const noexcept {
        return *p_impl;
    }

    boost::optional<deg_t> width() const noexcept;
    boost::optional<deg_t> depth() const noexcept;
    boost::optional<deg_t> degree(key_type key) const noexcept;
    std::string key_to_string(key_type key) const noexcept;
    dimn_t size(int deg) const noexcept;
    dimn_t start_of_degree(int deg) const noexcept;
    boost::optional<key_type> lparent(key_type key) const noexcept;
    boost::optional<key_type> rparent(key_type key) const noexcept;
    key_type index_to_key(dimn_t idx) const noexcept;
    dimn_t key_to_index(key_type key) const noexcept;
    let_t first_letter(key_type) const noexcept;
    key_type key_of_letter(let_t letter) const noexcept;
    bool letter(key_type key) const noexcept;
};

} // namespace algebra
} // namespace esig
#endif//ESIG_PATHS_BASIS_H_
