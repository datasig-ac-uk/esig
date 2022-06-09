//
// Created by sam on 09/05/22.
//

#include <esig/algebra/coefficients.h>

#include <stdexcept>

namespace esig {
namespace algebra {


std::shared_ptr<data_allocator> allocator_for_coeff(coefficient_type ctype)
{
    switch (ctype) {
        case coefficient_type::sp_real:
            return std::shared_ptr<data_allocator>(new dtl::allocator_ext<std::allocator<float>>);
        case coefficient_type::dp_real:
            return std::shared_ptr<data_allocator>(new dtl::allocator_ext<std::allocator<double>>);
    }
    throw std::invalid_argument("bad ctype");
}
std::shared_ptr<data_allocator> allocator_for_key_coeff(coefficient_type ctype)
{
    switch (ctype) {
        case coefficient_type::sp_real:
            return std::shared_ptr<data_allocator>(new dtl::allocator_ext<std::allocator<std::pair<key_type, float>>>);
        case coefficient_type::dp_real:
            return std::shared_ptr<data_allocator>(new dtl::allocator_ext<std::allocator<std::pair<key_type, double>>>);
    }
    throw std::invalid_argument("bad ctype");
}

}// namespace algebra
}// namespace esig
