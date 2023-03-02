//
// Created by user on 30/08/22.
//

#include "esig/algebra/fallback_operations.h"


using namespace esig;
using namespace esig::algebra;

bool esig::algebra::dtl::fallback_equals(const algebra_iterator &lbegin,
                          const algebra_iterator &lend,
                          const algebra_iterator &rbegin,
                          const algebra_iterator &rend) noexcept
{
    if (lbegin == lend && rbegin == rend) {
        return true;
    }
    if (lbegin == lend) {
        for (auto rit = rbegin; rit != rend; ++rit) {
            if (rit->value().is_zero()) {
                return false;
            }
        }
        return true;
    }
    if (rbegin == rend) {
        for (auto lit = lbegin; lit != lend; ++lit) {
            if (lit->value().is_zero()) {
                return false;
            }
        }
        return true;
    }

    auto lit = lbegin;
    auto rit = rbegin;

    key_type lkey = lit->key();
    key_type rkey = rit->key();
    while (lit != lend && rit != rend) {
        if (lkey == rkey) {
            if (lit->value() != rit->value()) {
                return false;
            }
            ++lit;
            ++rit;
            lkey = lit->key();
            rkey = rit->key();
        } else if (lkey < rkey) {
            if (lit->value().is_zero()) {
                return false;
            }
            ++lit;
            lkey = lit->key();
        } else {
            if (rit->value().is_zero()) {
                return false;
            }
            ++rit;
            rkey = rit->key();
        }
    }

    for (; lit != lend; ++lit) {
        if (lit->value().is_zero()) {
            return false;
        }
    }
    for (; rit != rend; ++rit) {
        if (rit->value().is_zero()) {
            return false;
        }
    }
    return true;
}
