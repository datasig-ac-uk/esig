//
// Created by user on 09/12/22.
//

#include "py_lie_key.h"

#include <algorithm>
#include <sstream>

using namespace esig;
using namespace esig::python;

template<typename LeftFn, typename RightFn>
void walk_tree(const py_lie_letter *tree, LeftFn left_visitor, RightFn right_visitor) {
    const auto *left = tree;
    const auto *right = ++tree;

    left_visitor(*left);
    right_visitor(*right);
    if (left->is_offset()) {
        walk_tree(left + static_cast<dimn_t>(*left), left_visitor, right_visitor);
    }
    if (right->is_offset()) {
        walk_tree(right + static_cast<dimn_t>(*right), left_visitor, right_visitor);
    }
}

template<typename Fn>
void walk_tree(const py_lie_letter *tree, Fn visitor) {
    walk_tree(tree, visitor, visitor);
}

bool branches_equal(const py_lie_letter *lhs, const py_lie_letter *rhs) {
    if (!lhs->is_offset() && !rhs->is_offset()) {
        return static_cast<let_t>(*lhs) == static_cast<let_t>(*rhs);
    }

    if (lhs->is_offset() && rhs->is_offset()) {
        return branches_equal(
            lhs + static_cast<dimn_t>(*lhs),
            rhs + static_cast<dimn_t>(*rhs));
    }

    return false;
}

struct print_walker {

    using pointer = const py_lie_letter *;
    std::stringstream ss;

    print_walker() : ss() {}

    void walk_single(pointer arg) {
        if (arg->is_offset()) {
            auto offset = static_cast<dimn_t>(*arg);
            walk_pair(arg + offset, arg + offset + 1);
        } else {
            ss << static_cast<let_t>(*arg);
        }
    }

    void walk_pair(pointer left, pointer right) {
        ss << '[';
        walk_single(left);
        ss << ',';
        walk_single(right);
        ss << ']';
    }

    std::string operator()(pointer tree) {
        walk_pair(tree, tree + 1);
        return ss.str();
    }
};

typename py_lie_key::container_type trim_branch(const boost::container::small_vector_base<py_lie_letter> &tree, dimn_t start) {
    assert(start == 0 || start == 1);
    if (tree.empty() || (tree.size() == 1 && start == 0)) {
        return {};
    }
    if (tree.size() == 1 && start == 1) {
        return {tree[0]};
    }
    if (tree.size() == 2) {
        return {tree[start]};
    }

    if (!tree[start].is_offset()) {
        return {tree[start]};
    }

    typename py_lie_key::container_type new_tree;
    new_tree.reserve(tree.size());
    dimn_t current = 0;
    dimn_t size = 0;

    auto visitor = [&new_tree, &current, &size](const py_lie_letter &node) {
      ++current;
      if (node.is_offset()) {
          // Each offset points to a pair further down the tree
          size += 2;
          // point to the first
          new_tree.emplace_back(py_lie_letter::from_offset(size - current));
      } else {
          new_tree.emplace_back(node);
      }
      ++size;
    };

    walk_tree(tree.data() + start + static_cast<dimn_t>(tree[start]), visitor);

    new_tree.shrink_to_fit();
    return new_tree;
}

py_lie_key::py_lie_key(deg_t width, const boost::container::small_vector_base<py_lie_letter> &data)
    : m_data(data), m_width(width) {
}
py_lie_key::py_lie_key(deg_t width, let_t left, let_t right)
    : m_data{py_lie_letter::from_letter(left), py_lie_letter::from_letter(right)}, m_width(width) {
    assert(left < right);
}
py_lie_key::py_lie_key(deg_t width, let_t left, const py_lie_key &right)
    : m_data{py_lie_letter::from_letter(left)}, m_width(width) {
    assert(m_width == right.m_width);
    m_data.assign(right.m_data.begin(), right.m_data.end());
    assert(!right.is_letter() || right.as_letter() > left);
}

py_lie_key::py_lie_key(esig::deg_t width)
    : m_width(width) {}
py_lie_key::py_lie_key(deg_t width, let_t letter)
    : m_data{py_lie_letter::from_letter(letter)}, m_width(width) {
    assert(0 < letter && letter <= width);
}
py_lie_key::py_lie_key(deg_t width, const py_lie_key &left, const py_lie_key &right)
    : m_data{py_lie_letter::from_offset(2), py_lie_letter::from_offset(1 + left.degree())},
      m_width(left.m_width) {
    m_data.assign(left.m_data.begin(), left.m_data.end());
    m_data.assign(right.m_data.begin(), right.m_data.end());
}
std::string py_lie_key::to_string() const {
    if (m_data.size() == 1) {
        std::stringstream ss;
        ss << static_cast<let_t>(m_data[0]);
        return ss.str();
    }
    print_walker walker;
    return walker(m_data.data());
}
py_lie_key py_lie_key::lparent() const {
    return py_lie_key(m_width, trim_branch(m_data, 0));
}
py_lie_key py_lie_key::rparent() const {
    return py_lie_key(m_width, trim_branch(m_data, 1));
}
deg_t py_lie_key::degree() const {
    return std::count_if(m_data.begin(), m_data.end(), [](const py_lie_letter &letter) { return !letter.is_offset(); });
}
bool py_lie_key::equals(const py_lie_key &other) const noexcept {
    const auto *lptr = m_data.data();
    const auto *rptr = other.m_data.data();
    return branches_equal(lptr, rptr) && branches_equal(++lptr, ++rptr);
}

bool py_lie_key::is_letter() const noexcept {
    return (m_data.size() == 1);
}
let_t py_lie_key::as_letter() const {
    assert(m_data.size() == 1);
    return static_cast<let_t>(m_data[0]);
}




namespace {

struct to_lie_key_helper {
    using container_type = typename ::py_lie_key::container_type;
    esig::dimn_t size;
    esig::dimn_t current;
    esig::deg_t width;
    esig::let_t max_letter = 0;

    explicit to_lie_key_helper(esig::deg_t w) : size(2), current(0), width(w) {}

    container_type parse_list(const py::handle &obj) {
        if (py::len(obj) != 2) {
            throw py::value_error("list items must contain exactly two elements");
        }

        auto left = obj[py::int_(0)];
        auto right = obj[py::int_(1)];

        return parse_pair(left, right);
    }

    container_type parse_single(const py::handle &obj) {
        py::handle result;
        if (py::isinstance<py::list>(obj)) {
            return parse_list(obj);
        }
        if (py::isinstance<py::int_>(obj)) {
            auto as_let = obj.cast<esig::let_t>();
            if (as_let > max_letter) {
                max_letter = as_let;
            }
            return container_type{py_lie_letter::from_letter(as_let)};
        }
        throw py::type_error("items must be either int or lists");
    }

    container_type parse_pair(const py::handle &left, const py::handle &right) {

        auto left_tree = parse_single(left);
        auto right_tree = parse_single(right);

        container_type result;
        esig::dimn_t left_size;

        result.reserve(2 + left_tree.size() + right_tree.size());
        if (left_tree.size() == 1) {
            left_size = 0;
            result.push_back(left_tree[0]);
        } else {
            left_size = left_tree.size();
            result.push_back(py_lie_letter::from_offset(2));
        }
        if (right_tree.size() == 1) {
            result.push_back(right_tree[0]);
        } else {
            result.push_back(py_lie_letter::from_offset(1 + left_size));
        }
        if (left_tree.size() > 1) {
            result.insert(result.end(), left_tree.begin(), left_tree.end());
        }
        if (right_tree.size() > 1) {
            result.insert(result.end(), right_tree.begin(), right_tree.end());
        }

        return result;
    }

    container_type operator()(const py::handle &obj) {
        if (!py::isinstance<py::list>(obj)) {
            throw py::type_error("expected a list with exactly two elements");
        }
        if (py::len(obj) != 2) {
            throw py::value_error("expected list with exactly 2 elements");
        }
        return parse_pair(obj[py::int_(0)], obj[py::int_(1)]);
    }

    esig::deg_t get_width() {
        if (width != 0 && max_letter > width) {
            throw py::value_error("a letter exceeds the width");
        } else {
            width = max_letter;
        }
        return width;
    }
};

py_lie_key make_lie_key(const py::args &args, const py::kwargs &kwargs) {
    esig::deg_t width = 0;

    if (kwargs.contains("width")) {
        width = kwargs["width"].cast<esig::deg_t>();
    }

    if (args.empty()) {
        throw py::value_error("argument cannot be empty");
    }

    if (py::isinstance<py::int_>(args[0])) {
        auto letter = args[0].cast<esig::let_t>();
        if (width != 0 && letter > width) {
            throw py::value_error("letter exceeds width");
        } else {
            width = esig::deg_t(letter);
        }
        return ::py_lie_key(width, letter);
    }

    if (!py::isinstance<py::list>(args[0])) {
        throw py::type_error("expected int or list");
    }

    to_lie_key_helper helper(width);

    return py_lie_key(helper.get_width(), helper(args[0]));
}

}// namespace

void esig::python::init_py_lie_key(pybind11::module_ &m) {
    py::class_<py_lie_key> klass(m, "LieKey");
    klass.def(py::init(&make_lie_key));

    klass.def("__str__", &py_lie_key::to_string);
}
