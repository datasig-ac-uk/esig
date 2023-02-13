//
// Created by user on 08/12/22.
//

#include "py_algebra.h"

#include "free_tensor.h"
#include "lie.h"
#include "py_lie_key.h"
#include "py_lie_key_iterator.h"
#include "py_tensor_key.h"
#include "py_tensor_key_iterator.h"
#include "py_context.h"


void esig::python::init_algebra(py::module_ &m) {

    init_py_lie_key(m);
    init_py_tensor_key(m);
    init_lie_key_iterator(m);
    init_tensor_key_iterator(m);

    init_lie(m);
    init_free_tensor(m);
    init_context(m);

}
