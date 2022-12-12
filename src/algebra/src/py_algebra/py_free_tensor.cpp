//
// Created by sam on 03/05/22.
//

#include "py_free_tensor.h"
#include <sstream>
#include <esig/algebra/context.h>
#include "py_iterator.h"
#include "convert_buffer.h"
#include <esig/pycommon.h>

#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace pybind11::literals;

using esig::deg_t;
using esig::dimn_t;
using esig::algebra::vector_type;

static const char* FREE_TENSOR_DOC = R"eadoc(Element of the (truncated) tensor algebra.

A :class:`tensor` object supports arithmetic operators, providing both objects are compatible,
along with comparison operators. The multiplication operator for this class is the free tensor
multiplication (concatenation of tensor words). Moreover, :class:`tensor` objects are
`iterable <https://docs.python.org/3/glossary.html#term-iterable>`_, where the items are tuples
of :class:`tensor_key` and :class:`float` corresponding to the non-zero elements of the
:class:`tensor`.

The class also supports (implicit and explicit) conversion to a Numpy array type, so it can be
used as an argument to any function that takes Numpy arrays. The array representation of a
:class:`tensor` is one-dimensional. Alternatively, one can construct d-dimensional arrays
containing the elements of degree d by using the :py:meth:`~tensor.degree_array` method.

There are methods for computing the tensor :py:meth:`~tensor.exponential`,
:py:meth:`~tensor.logarithm`, and :py:meth:`~tensor.inverse` (of group-like elements). See the
documentation of these methods for more information.

A tensor can be created from an array-like object containing the coefficients of the keys, in
their standard order. Since tensors must be created with both an alphabet size and depth, we need
to provide at least the ``depth`` argument. However, it is recommended that you also provided
the ``width`` argument, otherwise it is assumed that the tensor has degree 1 and the alphabet
size will be determined from the length of the argument.

.. code:: python

    >>> ts1 = esig_paths.tensor([1.0, 2.0, 3.0], depth=2)
    >>> print(ts1)
    { 1() 2(1) 3(2) }
    >>> ts2 = esig_paths.tensor([1.0, 2.0, 3.0], width=2, depth=2)
    >>> print(ts2)
    { 1() 2(1) 3(2) }

If the width argument is provided, this construction can be used to construct tensors of any
degree, up to the maximum. The :class:`~esig_paths.algebra_context` class provides a method
:py:meth:`~esig_paths.algebra_context.tensor_size` that can be used to get the dimension of the
tensor algebra_old up to a given degree.
)eadoc";

//temp
template <typename I, typename B>
constexpr B log(I arg, B base) noexcept
{
    return (arg < static_cast<I>(base)) ? 0 : (1 + log(arg/static_cast<I>(base), base));
}



esig::algebra::free_tensor ft_from_buffer(const py::object& arg, const py::kwargs& kwargs)
{

    auto helper = esig::algebra::get_construction_data(arg, kwargs);


    if (!helper.ctx) {
        // There isn't a lot we can do here. Any attempt to find
        // both width and depth is doomed to fail, so let's assume
        // the size of the buffer is 1 + W.
        if (helper.count == 0) {
            throw py::value_error("Cannot construct empty tensor without width/depth or context arguments");
        }

        helper.width = helper.count - 1;
        helper.ctx = esig::algebra::get_context(helper.width, helper.depth, helper.ctype);
    }

//    esig::algebra::vector_construction_data data(
//        helper.begin_ptr,
//        helper.end_ptr,
//        helper.ctype,
//        helper.vtype,
//        helper.input_vec_type,
//        helper.itemsize
//    );
//    return helper.ctx->construct_tensor(data);
    return helper.ctx->zero_tensor(vector_type::sparse);
}

void init_free_tensor_iterator(py::module& m)
{
    using esig::algebra::py_free_tensor_iterator;
    py::class_<py_free_tensor_iterator> klass(m, "FreeTensorIterator");

    klass.def("__next__", &py_free_tensor_iterator::next);
}


void esig::algebra::init_free_tensor(pybind11::module_ &m)
{
    init_free_tensor_iterator(m);

    pybind11::class_<free_tensor> klass(m, "FreeTensor", FREE_TENSOR_DOC);

    klass.def(py::init(&ft_from_buffer), "data"_a=py::none());

    klass.def_property_readonly("width", &free_tensor::width);
    klass.def_property_readonly("max_degree", &free_tensor::depth);
    klass.def_property_readonly("dtype", &free_tensor::coeff_type);
    klass.def_property_readonly("storage_type", &free_tensor::storage_type);

    klass.def("size", &free_tensor::size);
    klass.def("degree", &free_tensor::degree);

    klass.def("__getitem__", [](const free_tensor& self, key_type key) {
        return self[key];
    });
    klass.def("__iter__", [](const free_tensor& self) {
             return py_free_tensor_iterator(self.begin(), self.end());
         });


    klass.def("__neg__", &free_tensor::uminus, py::is_operator());

    klass.def("__add__", &free_tensor::add, py::is_operator());
    klass.def("__sub__", &free_tensor::sub, py::is_operator());
    klass.def("__mul__", &free_tensor::smul, py::is_operator());
    klass.def("__truediv__", &free_tensor::smul, py::is_operator());
    klass.def("__mul__", &free_tensor::mul, py::is_operator());
    klass.def("__rmul__", [](const free_tensor& self, const scalars::scalar& other) { return self.smul(other); },
            py::is_operator());

    klass.def("__mul__", [](const free_tensor& self, scalar_t arg) {
             return self.smul( scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__rmul__", [](const free_tensor& self, scalar_t arg) {
             return self.smul( scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__mul__", [](const free_tensor& self, long long arg) {
             return self.smul( scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__rmul__", [](const free_tensor& self, long long arg) {
             return self.smul( scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__truediv__", [](const free_tensor& self, scalar_t arg) {
             return self.sdiv( scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__truediv__", [](const free_tensor& self, long long arg) {
             return self.sdiv( scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());

    klass.def("__iadd__", &free_tensor::add_inplace, py::is_operator());
    klass.def("__isub__", &free_tensor::sub_inplace, py::is_operator());
    klass.def("__imul__", &free_tensor::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &free_tensor::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &free_tensor::mul_inplace, py::is_operator());

    klass.def("__imul__", [](free_tensor& self, scalar_t arg) {
             return self.smul_inplace( scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__imul__", [](free_tensor& self, long long arg) {
             return self.smul_inplace( scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__itruediv__", [](free_tensor& self, scalar_t arg) {
             return self.sdiv_inplace( scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__itruediv__", [](free_tensor& self, long long arg) {
             return self.sdiv_inplace( scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());

    klass.def("add_scal_mul", &free_tensor::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &free_tensor::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &free_tensor::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &free_tensor::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &free_tensor::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &free_tensor::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &free_tensor::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &free_tensor::mul_sdiv, "other"_a, "scalar"_a);

    klass.def("exp", &free_tensor::exp);
    klass.def("log", &free_tensor::log);
    klass.def("inverse", &free_tensor::inverse);
    klass.def("fmexp", &free_tensor::fmexp, "other"_a);

    klass.def("__str__", [](const free_tensor& self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
    });

    klass.def("__repr__", [](const free_tensor& self) {
                std::stringstream ss;
                ss << "FreeTensor(width=" << self.width() << ", depth=" << self.depth();
                ss << ", ctype=" << self.coeff_type()->info().name << ')';
                return ss.str();
            });

    klass.def("__eq__", [](const free_tensor& lhs, const free_tensor& rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const free_tensor& lhs, const free_tensor& rhs) { return lhs != rhs; });

    klass.def("__array__", [](const free_tensor& self) {
//        py::dtype dtype = dtype_from(self.coeff_type());
        py::dtype dtype = ::esig::ctype_to_npy_dtype(self.coeff_type());

        if (self.storage_type() == vector_type::dense) {
            auto it = self.iterate_dense_components().next();
            if (!it) {
                throw std::runtime_error("dense data should be valid");
            }
            const auto* ptr = it.begin();
            return py::array(dtype, {self.size()}, {}, ptr);
        }
        return py::array(dtype);
    });

}
