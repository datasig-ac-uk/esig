//
// Created by user on 29/04/22.
//

#include <esig/libalgebra_context/libalgebra_context.h>

#include "libalgebra_context_maker.h"
#include <mutex>


void esig::algebra::install_default_libalgebra_contexts()
{
    static std::atomic_bool done(false);

    if (!done) {
        std::unique_ptr<esig::algebra::context_maker> maker(
                new esig::algebra::libalgebra_context_maker()
                );
        esig::algebra::register_context_maker(std::move(maker));
        done.store(true);
    }

}
