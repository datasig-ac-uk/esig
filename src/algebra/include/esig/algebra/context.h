//
// Created by user on 20/03/2022.
//

#ifndef ESIG_ALGEBRA_CONTEXT_H_
#define ESIG_ALGEBRA_CONTEXT_H_

#include <esig/algebra/context_fwd.h>


namespace esig {
namespace algebra {


struct ESIG_ALGEBRA_EXPORT data_iterator {
    virtual ~data_iterator() = default;

    virtual const char *dense_begin() = 0;
    virtual const char *dense_end() = 0;

    virtual bool next_sparse() = 0;
    virtual const void *sparse_kv_pair() = 0;

    virtual bool advance() = 0;
    virtual bool finished() const = 0;
};


template<typename ToLie>
struct increment_iterable {

    struct iterator {
        using value_type = decltype(std::declval<ToLie>()(nullptr, nullptr));
        using reference = value_type;
        using pointer = value_type *;

    private:
        ToLie &m_to_lie;
        data_iterator *m_ptr;
        coefficient_type m_ctype;
        vector_type m_vtype;

    public:
        explicit iterator(
                coefficient_type ctype,
                vector_type vtype,
                data_iterator *iter,
                ToLie &to_lie);

        reference operator*() const;
        //        pointer operator->() const;

        iterator &operator++();

        bool operator==(const iterator &other) const;
        bool operator!=(const iterator &other) const;
    };

private:
    data_iterator *m_ptr;
    coefficient_type m_ctype;
    vector_type m_vtype;
    ToLie m_to_lie;

public:
    increment_iterable(
            coefficient_type ctype,
            vector_type vtype,
            data_iterator *p,
            ToLie &&to_lie);

    iterator begin();
    iterator end();
};

class ESIG_ALGEBRA_EXPORT signature_data
{
    coefficient_type m_coeff_type;
    vector_type m_vector_type;

    std::unique_ptr<data_iterator> m_iter;

public:
    template<typename Iter>
    explicit signature_data(
            coefficient_type ctype,
            vector_type vtype,
            Iter &&iter);

    coefficient_type ctype() const;
    vector_type vtype() const;


    template<typename ToLie>
    increment_iterable<ToLie> iter_increments(ToLie &&to_lie)
    {
        return increment_iterable<ToLie>(m_coeff_type, m_vector_type, m_iter.get(),
                                         std::forward<ToLie>(to_lie));
    }
};


struct ESIG_ALGEBRA_EXPORT derivative_compute_info {
    lie logsig_of_interval;
    lie perturbation;
};


class ESIG_ALGEBRA_EXPORT vector_construction_data
{

public:
    vector_construction_data(
            coefficient_type ctype,
            vector_type vtype);

    vector_construction_data(
            const char *data_begin,
            const char *data_end,
            coefficient_type ctype,
            vector_type vtype,
            input_data_type idtype,
            dimn_t itemsize);

    template<typename Scalar>
    vector_construction_data(
            const Scalar *data_begin,
            const Scalar *data_end,
            vector_type vtype);

    template<typename Scalar>
    vector_construction_data(
            const std::pair<key_type, Scalar> *data_begin,
            const std::pair<key_type, Scalar> *data_end,
            vector_type vtype);

    vector_construction_data(
            const coefficient *data_begin,
            const coefficient *data_end,
            vector_type vtype);

    vector_construction_data(
            const std::pair<key_type, coefficient> *data_begin,
            const std::pair<key_type, coefficient> *data_end,
            vector_type vtype);

    const char *begin() const;
    const char *end() const;
    coefficient_type ctype() const;
    vector_type vtype() const;
    input_data_type input_type() const;
    dimn_t item_size() const;
    coefficient_type m_coeffs;
    const char *m_data_begin;
    const char *m_data_end;
    input_data_type m_data_type;
    dimn_t m_item_size;
    vector_type m_vect_type;
};


/**
 * @brief Context object that controls creation and manipulation of algebra types.
 *
 *
 */
class ESIG_ALGEBRA_EXPORT context
{
public:
    using tensor_r = free_tensor_interface;
    using lie_r = lie_interface;

    using tensor_basis_r = algebra_basis;
    using lie_basis_r = algebra_basis;

    virtual ~context() = default;

    // Get information about the context
    virtual deg_t width() const noexcept = 0;
    virtual deg_t depth() const noexcept = 0;
    virtual coefficient_type ctype() const noexcept = 0;

    // Get related contexts
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth) const = 0;
    virtual std::shared_ptr<const context> get_alike(coefficient_type new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth, coefficient_type new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const = 0;

    // Scalar allocators
    virtual std::shared_ptr<data_allocator> coefficient_alloc() const = 0;
    virtual std::shared_ptr<data_allocator> pair_alloc() const = 0;

    // Get information about the tensor and lie types
    virtual dimn_t lie_size(deg_t d) const noexcept = 0;
    virtual dimn_t tensor_size(deg_t d) const noexcept = 0;

    virtual bool check_compatible(const context& other) const noexcept;

    virtual free_tensor convert(const free_tensor& arg, vector_type new_vec_type) const = 0;
    virtual lie convert(const lie& arg, vector_type new_vec_type) const = 0;

    //TODO: needs more thought

    // Access the basis interface for lie and tensors
    //virtual std::shared_ptr<algebra_basis> get_tensor_basis() const noexcept = 0;
    //virtual std::shared_ptr<algebra_basis> get_lie_basis() const noexcept = 0;
//    virtual const algebra_basis &borrow_tbasis() const noexcept = 0;
//    virtual const algebra_basis &borrow_lbasis() const noexcept = 0;

    virtual basis get_tensor_basis() const = 0;
    virtual basis get_lie_basis() const = 0;


    // Construct new instances of tensors and lies
    virtual free_tensor construct_tensor(const vector_construction_data &) const = 0;
    virtual lie construct_lie(const vector_construction_data &) const = 0;
    virtual free_tensor zero_tensor(vector_type vtype) const;
    virtual lie zero_lie(vector_type vtype) const;

    // Conversions between Lie and Tensors
    virtual free_tensor lie_to_tensor(const lie &arg) const = 0;
    virtual lie tensor_to_lie(const free_tensor &arg) const = 0;

    // Campbell-Baker-Hausdorff formula
    virtual lie cbh(const std::vector<lie> &lies, vector_type vtype) const = 0;


    // Methods for computing signatures
    virtual free_tensor to_signature(const lie &log_sig) const = 0;
    virtual free_tensor signature(signature_data data) const = 0;
    virtual lie log_signature(signature_data data) const = 0;

    // Method for computing the derivative of a signature
    virtual free_tensor sig_derivative(
            const std::vector<derivative_compute_info> &info,
            vector_type vtype,
            vector_type) const = 0;
};


std::shared_ptr<const context> ESIG_ALGEBRA_EXPORT get_context(
        deg_t width,
        deg_t depth,
        coefficient_type ctype,
        const std::vector<std::string> &preferences = {});


struct ESIG_ALGEBRA_EXPORT context_maker {
    virtual ~context_maker() = default;
    virtual bool can_get(deg_t, deg_t, coefficient_type) const noexcept = 0;
    virtual int get_priority(const std::vector<std::string> &preferences) const noexcept;
    virtual std::shared_ptr<const context> get_context(deg_t, deg_t, coefficient_type) const = 0;
};

ESIG_ALGEBRA_EXPORT
const context_maker *register_context_maker(std::unique_ptr<context_maker> maker);


template<typename Maker>
struct register_maker_helper {
    const context_maker *maker;

    register_maker_helper()
    {
        maker = register_context_maker(std::unique_ptr<context_maker>(new Maker()));
    }

    template<typename... Args>
    explicit register_maker_helper(Args &&...args)
    {
        maker = register_context_maker(std::unique_ptr<context_maker>(new Maker(std::forward<Args>(args)...)));
    }
};


// Implementation of template methods

template<typename Iter>
signature_data::signature_data(
        coefficient_type ctype,
        enum vector_type vtype,
        Iter &&iter)
    : m_coeff_type(ctype), m_vector_type(vtype), m_iter(new Iter(std::forward<Iter>(iter)))
{}

template<typename ToLie>
increment_iterable<ToLie>::increment_iterable(
        coefficient_type ctype,
        vector_type vtype,
        data_iterator *p,
        ToLie &&to_lie)
    : m_ptr(p), m_ctype(ctype), m_vtype(vtype), m_to_lie(std::forward<ToLie>(to_lie))
{
}
template<typename ToLie>
typename increment_iterable<ToLie>::iterator increment_iterable<ToLie>::begin()
{
    return increment_iterable::iterator(m_ctype, m_vtype, m_ptr, m_to_lie);
}

template<typename ToLie>
typename increment_iterable<ToLie>::iterator increment_iterable<ToLie>::end()
{
    return increment_iterable::iterator(m_ctype, m_vtype, nullptr, m_to_lie);
}


template<typename ToLie>
increment_iterable<ToLie>::iterator::iterator(
        coefficient_type ctype,
        vector_type vtype,
        data_iterator *iter,
        ToLie &to_lie)
    : m_to_lie(to_lie), m_ptr(iter), m_vtype(vtype), m_ctype(ctype)
{
}

template<typename ToLie>
typename increment_iterable<ToLie>::iterator::reference increment_iterable<ToLie>::iterator::operator*() const
{
    if (m_ptr == nullptr) {
        throw std::runtime_error("Increment iterator cannot be null");
    }
    if (m_vtype == vector_type::dense) {
        return m_to_lie(m_ptr->dense_begin(), m_ptr->dense_end());
    }
    if (m_vtype == vector_type::sparse) {
        return m_to_lie(m_ptr);
    }
    throw std::runtime_error("Unrecognised vector_type");
}

template<typename ToLie>
typename increment_iterable<ToLie>::iterator &increment_iterable<ToLie>::iterator::operator++()
{
    if (m_ptr == nullptr) {
        throw std::runtime_error("advancing invalid iterable");
    }
    m_ptr->advance();
    return *this;
}

template<typename ToLie>
bool increment_iterable<ToLie>::iterator::operator==(const increment_iterable::iterator &other) const
{
    return m_ptr != nullptr && m_ptr->finished();
}

template<typename ToLie>
bool increment_iterable<ToLie>::iterator::operator!=(const increment_iterable::iterator &other) const
{
    return !operator==(other);
}


template<typename Scalar>
vector_construction_data::vector_construction_data(const Scalar *data_begin, const Scalar *data_end, vector_type vtype)
    : m_data_begin(reinterpret_cast<const char *>(data_begin)),
      m_data_end(reinterpret_cast<const char *>(data_end)),
      m_coeffs(dtl::get_coeff_type(Scalar(0))),
      m_vect_type(vtype),
      m_data_type(input_data_type::value_array),
      m_item_size(sizeof(Scalar))
{
}

template<typename Scalar>
vector_construction_data::vector_construction_data(const std::pair<key_type, Scalar> *data_begin, const std::pair<key_type, Scalar> *data_end, vector_type vtype)
    : m_data_begin(reinterpret_cast<const char *>(data_begin)),
      m_data_end(reinterpret_cast<const char *>(data_end)),
      m_coeffs(dtl::get_coeff_type(Scalar(0))),
      m_vect_type(vtype),
      m_data_type(input_data_type::key_value_array),
      m_item_size(sizeof(std::pair<key_type, Scalar>))
{
}


}// namespace algebra
}// namespace esig


#endif//ESIG_ALGEBRA_CONTEXT_H_
