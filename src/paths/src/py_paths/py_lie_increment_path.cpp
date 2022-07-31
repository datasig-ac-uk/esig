//
// Created by sam on 02/05/22.
//

#include "py_lie_increment_path.h"
#include <esig/paths/python_interface.h>
#include "python_arguments.h"
#include "pybind11/numpy.h"

#include <cassert>

using esig::paths::path;
using esig::deg_t;
using esig::dimn_t;
using esig::param_t;
using esig::key_type;
using esig::algebra::vector_type;
using esig::algebra::coefficient_type;
namespace py = pybind11;

void esig::paths::init_lie_increment_path(py::module_ &m)
{
    py::class_<esig::paths::lie_increment_path, esig::paths::path_interface> klass(m, "LieIncrementPath");

    esig::paths::register_pypath_constructor(py::type::of<esig::paths::lie_increment_path>(),
            construct_lie_increment_path);
}


namespace {

template <coefficient_type CType>
struct buffer_creator_helper
{
    using scalar_type = esig::algebra::type_of_coeff<CType>;

    explicit buffer_creator_helper(deg_t width) : m_path_width(width)
    {}

    void operator()(const py::buffer_info& buffer_info)
    {
        const auto& shape = buffer_info.shape;
        const auto& strides = buffer_info.strides;
        const char* ptr = reinterpret_cast<const char*>(buffer_info.ptr);

        auto length = shape[0];
        auto current_width = shape[1];
        assert(current_width <= m_path_width);

        auto new_size = tmp_buffer.size() + length*m_path_width;
        tmp_buffer.reserve(new_size);

        if (!strides.empty()) {
            auto get = [=](dimn_t i, dimn_t j) {
              const char* p = ptr + i*strides[0] + j*strides[1];
              return *reinterpret_cast<const scalar_type*>(p);
            };

            for (dimn_t sample=0; sample<length; ++sample) {
                for (dimn_t col=0; col<current_width; ++col) {
                    tmp_buffer.push_back(get(sample, col));
                }
                for (dimn_t col=current_width; col<m_path_width; ++col) {
                    tmp_buffer.push_back(scalar_type(0));
                }
            }
        } else {
            const auto* data_p = reinterpret_cast<const scalar_type*>(ptr);

            for (dimn_t sample=0; sample<length; ++sample) {
                for (dimn_t col=0; col<current_width; ++col) {
                    tmp_buffer.push_back(*(data_p++));
                }
                for (dimn_t col = current_width; col < m_path_width; ++col) {
                    tmp_buffer.push_back(scalar_type(0));
                }
            }
        }

        assert(tmp_buffer.size() == new_size);
    }

    esig::algebra::rowed_data_buffer
    to_buffer(const esig::algebra::context& ctx)
    {
        esig::algebra::rowed_data_buffer result(ctx.coefficient_alloc(), m_path_width, length());
        std::uninitialized_copy(tmp_buffer.begin(), tmp_buffer.end(),
                                reinterpret_cast<scalar_type*>(result.begin()));
        return result;
    }

    dimn_t length() const noexcept
    {
        return tmp_buffer.size() / m_path_width;
    }

private:
    deg_t m_path_width;
    std::vector<scalar_type> tmp_buffer;
};


template <coefficient_type CType>
path construct_path_impl(const py::args& args, const py::kwargs& kwargs)
{
    // We've already checked that len(args) >= 1
    auto first_buffer = args[0].cast<py::buffer>();
    auto first_buffer_info = first_buffer.request();

    auto ndim = first_buffer_info.ndim;
    if (ndim < 2 || ndim > 2) {
        throw py::value_error("unexpected number of data dimensions, expected 2");
    }
    const auto& shape = first_buffer_info.shape;
    auto width = shape[1];

    buffer_creator_helper<CType> helper(width);


    for (auto item : args) {
        auto buf_info = item.cast<py::buffer>().request();
        helper(buf_info);
    }

    auto depth = kwargs["depth"].cast<deg_t>();
    auto ctx = esig::algebra::get_context(width, depth, CType, {});


    esig::real_interval domain;

    esig::paths::additional_args additional;

    auto md = esig::paths::parse_kwargs_to_metadata(kwargs, additional);

    std::vector<param_t> indices;

    if (kwargs.contains("indices")) {
        const auto pyindices = kwargs["indices"].cast<py::array_t<param_t, py::array::forcecast>>();

        const param_t* ptr = pyindices.data();
        indices = std::vector<param_t>(ptr, ptr+pyindices.size());
        if (indices.size() != helper.length()) {
            throw py::value_error("indices length must match length of paths");
        }
    } else {
        auto len = helper.length();
        indices.reserve(len);

        for (dimn_t i=0; i<len; ++i) {
            indices.emplace_back(i);
        }
    }


    return path(esig::paths::lie_increment_path(
        helper.to_buffer(*ctx),
        indices,
        md
        ));
}


std::vector<param_t> process_indices(const py::handle& indices)
{
    std::vector<param_t> buffer;

    if (py::isinstance<py::buffer>(indices)) {
        auto info = indices.cast<py::buffer>().request();

        if (info.ndim != 1) {
            throw py::value_error("expected one dimensional array of floating points numbers");
        }

        switch (info.format[0]) {
            case 'f': {
                buffer.reserve(info.size);
                const auto *ptr = reinterpret_cast<const float *>(info.ptr);
                for (dimn_t i = 0; i < info.size; ++i) {
                    buffer.emplace_back(ptr[i]);
                }
                break;
            }
            case 'd': {
                const auto *ptr = reinterpret_cast<const double *>(info.ptr);
                buffer.assign(ptr, ptr + info.size);
                break;
            }
            default:
                throw py::type_error("expected array of floats or doubles");
        }
    } else if (py::isinstance<py::iterable>(indices)) {
        buffer.reserve(len_hint(indices));

        for (const auto& item : indices) {
            buffer.emplace_back(item.cast<param_t>());
        }
    } else {
        throw py::type_error("expected array or iterable of numbers");
    }

    return buffer;
}



void copy_py_buffer_to_data_buffer_helper(esig::algebra::data_buffer& buffer, const py::buffer_info& info, coefficient_type ctype)
{
#define ESIG_SWITCH_FN(CTYPE) esig::paths::copy_py_buffer_to_data_buffer<CTYPE>(buffer.begin(), info);
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}



} // namespace





esig::paths::path esig::paths::construct_lie_increment_path(const pybind11::args &args, const pybind11::kwargs &kwargs)
{

    if (args.empty()) {
        throw py::value_error("At least one argument must be provided");
    }
    if (args.size() != 1) {
        throw py::value_error("expected a single argument containing path data");
    }

    additional_args parse_args;
    esig::algebra::rowed_data_buffer data_buffer;
    std::vector<param_t> indices_buffer;
    path_metadata md;

    if (kwargs.contains("indices")) {
        indices_buffer = process_indices(kwargs["indices"]);
        auto minmax = std::minmax_element(indices_buffer.begin(), indices_buffer.end());
        parse_args.domain = real_interval {*minmax.first, *minmax.second};
    }

    if (py::isinstance<py::buffer>(args[0])) {
        const auto info = args[0].cast<py::buffer>().request();

        coefficient_type input_ctype;
        switch (info.format[0]) {
            case 'f':
                parse_args.ctype = coefficient_type::sp_real;
                input_ctype = coefficient_type::sp_real;
                break;
            case 'd':
                parse_args.ctype = coefficient_type::dp_real;
                input_ctype = coefficient_type::dp_real;
                break;
            default:
                throw py::type_error("unsupported data type");
        }

        if (info.ndim != 2) {
            throw py::value_error("data must be 2 dimensional");
        }

        auto n_samples = info.shape[0];
        parse_args.width = deg_t(info.shape[1]);
        if (indices_buffer.empty()) {
            indices_buffer.reserve(n_samples);
            for (dimn_t i=0; i<n_samples; ++i) {
                indices_buffer.emplace_back(i);
            }
            parse_args.domain = real_interval {param_t(0), param_t(n_samples+1)};
        } else {
            if (indices_buffer.size() != n_samples) {
                throw py::value_error("mismatch in number of samples vs number of indices");
            }
        }

        md = esig::paths::parse_kwargs_to_metadata(kwargs, parse_args);

        auto alloc = esig::algebra::allocator_for_coeff(md.ctype);
        data_buffer = algebra::rowed_data_buffer(alloc, md.width, n_samples);
        copy_py_buffer_to_data_buffer_helper(data_buffer, info, md.ctype);

    }

    return path(lie_increment_path(std::move(data_buffer), indices_buffer, md));
}
