//
// Created by sam on 03/05/22.
//

#ifndef ESIG_PY_LIE_H
#define ESIG_PY_LIE_H

#include <pybind11/pybind11.h>
#include <esig/algebra/lie_interface.h>
#include <esig/algebra/python_interface.h>


namespace esig {
namespace algebra {


class py_lie_iterator
{
    algebra_iterator m_it;
    algebra_iterator m_end;

public:
    py_lie_iterator(algebra_iterator it, algebra_iterator end);
    std::pair<py_lie_key, coefficient> next();


};


void init_py_lie(pybind11::module_& m);

} // namespace algebra
} // namespace esig


#endif//ESIG_PY_LIE_H
