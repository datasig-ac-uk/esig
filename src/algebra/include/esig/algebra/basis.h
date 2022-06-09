//
// Created by sam on 07/03/2022.
//

#ifndef ESIG_PATHS_BASIS_H_
#define ESIG_PATHS_BASIS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>

#include <string>



namespace esig {
namespace algebra {


class ESIG_ALGEBRA_EXPORT algebra_basis
{
public:

    virtual ~algebra_basis();

    virtual deg_t width() const noexcept = 0;
    virtual deg_t depth() const noexcept = 0;

    virtual deg_t degree(const key_type&) const = 0;
    virtual std::string key_to_string(const key_type&) const = 0;
    virtual key_type key_of_letter(let_t) const noexcept = 0;
    virtual bool letter(const key_type&) const noexcept = 0;

    virtual dimn_t size(int) const noexcept = 0;
    virtual dimn_t start_of_degree(deg_t) const noexcept = 0;

    virtual key_type lparent(const key_type&) const noexcept = 0;
    virtual key_type rparent(const key_type&) const noexcept = 0;
    virtual let_t first_letter(const key_type&) const noexcept = 0;


};


} // namespace algebra
} // namespace esig
#endif//ESIG_PATHS_BASIS_H_
