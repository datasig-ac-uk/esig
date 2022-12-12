//
// Created by sam on 27/05/22.
//

#ifndef ESIG_PY_ITERATOR_H
#define ESIG_PY_ITERATOR_H

#include <esig/algebra/iteration.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>


namespace esig {
namespace algebra {


class py_algebra_iterator
{
    algebra_iterator m_it, m_end;

public:
    py_algebra_iterator(esig::algebra::algebra_iterator it, esig::algebra::algebra_iterator end);

    std::pair<key_type, scalars::scalar> next();


};

void init_py_iterator(pybind11::module_& m);


} // namespace algebra
} // namespace esig


#endif//ESIG_PY_ITERATOR_H
