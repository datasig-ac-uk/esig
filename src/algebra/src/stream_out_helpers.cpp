//
// Created by sam on 27/05/22.
//


#include <esig/algebra/coefficients.h>

using esig::algebra::coefficient_type;




std::ostream &esig::algebra::operator<<(std::ostream &os, const coefficient_type& ctype)
{
    switch (ctype) {
        case coefficient_type::dp_real:
            os << "DPReal";
        case coefficient_type::sp_real:
            os << "SPReal";
    }
    return os;
}
