//
// Created by sam on 30/03/2022.
//

#include "esig/paths/python_interface.h"
#include "py_path.h"
#include "py_lie_increment_path.h"
#include "py_function_path.h"


namespace py = pybind11;
using namespace pybind11::literals;

using namespace esig;


static const char *PATH_DOC = R"epdoc(Path object representing an abstract path.

In this context, a path is any object that can produce a log-signature over an interval. This
class is a wrapper around an arbitrary path type, and exposes methods for computing the
log-signature and signature over an interval. The underlying type of this path is not known after
an object is created. Both this class and the hidden implementation path type must derive from
:class:`~esig_paths .base_path` class, and so implement the functions for computing signatures
and log signatures.

Constructing an object of this class will create one of the underlying path object based on the
arguments provided:

    * an array-like object will create a tick path, where the rows in the array are treated as
      Lie increments;
    * a function that takes a single argument will create a function driven path, where
      increments are generated by taking the difference of the values given by the function at
      the ends of an interval.

A tick path is a map of parameter interval values to Lie increments that occur at that parameter.
For example, a tick path might contain the data ``{0.0: (1.0, 2.0), 1.0: (3.0, 4.0)}``, which
associates the dense_increment ``(1.0, 2.0)`` to the parameter value ``0.0`` and ``(3.0, 4.0)`` to the value
``1.0``. The path is considered constant between increments, and so a log-signature_impl calculated over
an interval that contains no increments will yield zero.

The required arguments for a tick path are the data, which should be an array-like object, and
the depth to which signatures/log-signatures should be computed. With these arguments, the first
column of the data array will be taken as the parameter values, and the remaining columns are
taken to be the corresponding Lie dense_increment. Optionally, you can also provide a separate array of
indices in which case all columns of the data array are taken to be Lie increments. You can also
specify the desired width. This ensures the width is set correctly to what you expect, but the
array must still be of the correct size.

For example, we can create the following tick paths-old, which will generate
equivalent paths-old

.. code:: python

    from esig.paths import path
    import numpy as np

    data = np.array([[1.0, 2.0], [3.0, 4.0]])  # the data
    indices = np.array([[0.0], [1.0]])

    p1 = path(np.concatenate((indices, data), axis=1), depth=2)
    p2 = path(data, indices=indices, depth=2)
    p3 = path(data, indices=indices, width=2, depth=2)

    assert p1.log_signature(0.0, 2.0, 0.0) == p2.log_signature(0.0, 2.0, 0.0)
    assert p2.log_signature_impl(0.0, 2.0, 0.0) == p3.log_signature_impl(0.0, 2.0, 0.0)

A function driven path is a path in which a function (or any callable object) that returns the
Lie values at a given parameter value as an array like object. As with tick data, you need to
specify the depth to which signatures and log-signatures should be calculated with the ``depth``
argument. The constructor will attempt to determine the width by evaluating the function at 0.0
and ``1.0`` if it is not provided using the ``width`` argument. If a width is specified, it will check
the consistency of the function output with this value. A simple example of a function path is
given below

.. code:: python

    from esig_paths import path

    def f(t):
        return (t, 2*t, t**2)

    p = path(f, depth=2)

    assert p.width == 3

You can also create function driven paths-old where the function takes a dyadic interval as an
argument and returns a dense_increment value (as an array-like object). To do this, set the
``increment_function`` argument to ``True``.

)epdoc";
static const char *domain_doc = R"epdoc(Get the domain interval of this path.

Returns the domain over which the path is defined. This will usually be the whole space on which
the path has values, but this can be modified by restricting the path to a smaller interval.

)epdoc";


static const char *SIG_DOC = R"epdoc(Compute the signature of the path object over an interval.

This function has three possible signatures, depending on how the interval is specified
(examples below):

    1. using a :class:`~esig.dyadic_interval`;
    2. using an :class:`~esig.interval`;
    3. by specifying the infimum and supremum of the interval as :class:`float` values.

In any case, an argument giving the desired accuracy of the calculation must be provided. This
behaviour of the parameter is different for different kinds of path. Broadly speaking, this
parameter determines the resolution of dyadic intervals that are used internally to calculate the
signature_impl. The resolution should be chosen by the underlying path type to so that the length of
the dyadic interval is not larger than the specified accuracy, although it need not be the first
such length.

)epdoc";


static const char *LSIG_DOC = R"epdoc(Compute the log signature over a dyadic interval.

:param dyadic_interval dyadic_interval: Dyadic interval over which the log signature should
    be calculated.
:param float accuracy: The accuracy to which the log signature should be computed.
    Has no effect for tick paths-old.
:returns: The log signature_impl of the path.
:rtype: esig.algebra.lie

)epdoc";

static const char* SIGDER_DOC = R"epdoc(


)epdoc";

static const char *restrict_doc = R"epdoc(Restrict the path to an interval.

:param interval interval: Interval to restrict path to.

)epdoc";

static const char *concatenate_doc = R"epdoc(Concatenate this path with another path on the right
 hand side.

:param path other: Right hand side path to concatenate to this path.
:returns: New path containing the concatenation of these paths-old.
:rtype: path

)epdoc";

static const char *init_tickpath_doc = R"epdoc(Create a new path from tick data.

)epdoc";

PYBIND11_MODULE(_paths, m) {

    py::options options;
    options.disable_function_signatures();

    py::class_<paths::path> path_class(m, "Path", PATH_DOC);

    path_class.def(py::init(&paths::py_path_constructor));

    auto interval_lsig =
            [](const paths::path &self, const interval &domain, accuracy_t acc) {
                return self.log_signature(domain, acc);
    };
    path_class.def("log_signature",
                   interval_lsig,
                   "interval"_a,
                   "accuracy"_a,
                   LSIG_DOC
                   );

    auto param_lsig = [](const paths::path &self,
                         param_t a, param_t b,
                         accuracy_t acc) {
        return self.log_signature(real_interval(a, b), acc);
    };
    path_class.def("log_signature",
                   param_lsig,
                   "lower"_a,
                   "upper"_a,
                   "accuracy"_a);

    auto interval_sig = [](const paths::path& self,
                           interval& domain,
                           accuracy_t acc){
        return self.signature(domain, acc);
    };
    path_class.def("signature",
                   interval_sig,
                   "interval"_a,
                   "accuracy"_a,
                   SIG_DOC);

    auto param_sig = [](const paths::path& self,
                        param_t a, param_t b,
                        accuracy_t acc) {
        auto sig = self.signature(real_interval(a, b), acc);
        py::print(sig);
        return sig;
    };
    path_class.def("signature",
                   std::move(param_sig),
                   "lower"_a,
                   "upper"_a,
                   "accuracy"_a);

    auto interval_sigder = [](const paths::path& self,
                              interval& domain,
                              const algebra::lie & perturbation,
                              accuracy_t acc) {
        return self.signature_derivative(domain, perturbation, acc);
    };
    path_class.def("signature_derivative",
                   interval_sigder,
                   "interval"_a,
                   "perturbation"_a,
                   "accuracy"_a,
                   SIGDER_DOC
                   );

    auto param_sigder = [](const paths::path& self,
                      param_t a, param_t b,
                      const algebra::lie & perturbation,
                      accuracy_t acc) {
        real_interval domain(a, b);
        return self.signature_derivative(domain, perturbation, acc);
    };
    path_class.def("signature_derivative",
                   param_sigder,
                   "lower"_a,
                   "upper"_a,
                   "perturbation"_a,
                   "accuracy"_a);

    auto interval_list_sigder = [](const paths::path& self,
                                   typename paths::path::perturbation_list_t perturbation_list,
                                   accuracy_t acc) {
        return self.signature_derivative(perturbation_list, acc);
    };
    path_class.def("signature_derivative",
                   interval_list_sigder,
                   "perturbations"_a,
                   "accuracy"_a);

    auto param_list_sigder = [](const paths::path& self,
                                std::vector<std::tuple<param_t, param_t, algebra::lie>> perturbation_list,
                                accuracy_t acc
                                ) {
        typename paths::path::perturbation_list_t perturbs;
        perturbs.reserve(perturbation_list.size());
        for (auto t : perturbation_list) {
            perturbs.emplace_back(
                    real_interval(std::get<0>(t), std::get<1>(t)),
                    std::get<2>(t)
            );
        }
        return self.signature_derivative(perturbs, acc);
    };
    path_class.def("signature_derivative",
                   param_list_sigder,
                   "perturbations"_a,
                   "accuracy"_a);

    esig::paths::init_python_path_interface(m);
}