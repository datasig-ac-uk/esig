//
// Created by user on 20/03/2022.
//

#ifndef ESIG_ALGEBRA_CONTEXT_H_
#define ESIG_ALGEBRA_CONTEXT_H_

#include <esig/algebra/context_fwd.h>

#include <esig/scalars.h>


namespace esig {
namespace algebra {



struct signature_data
{
    scalars::scalar_stream data_stream;
    std::vector<const key_type*> key_stream;
    vector_type vect_type;
};


struct derivative_compute_info {
    lie logsig_of_interval;
    lie perturbation;
};


struct vector_construction_data
{
    scalars::key_scalar_array data;
    vector_type vect_type = vector_type::sparse;
};


/**
 * @brief Context object that controls creation and manipulation of algebra types.
 *
 *
 */
class ESIG_ALGEBRA_EXPORT context
{
    const scalars::scalar_type* p_ctype;

protected:

    explicit context(const scalars::scalar_type* ctype) : p_ctype(ctype)
    {}

public:
    using tensor_r = free_tensor_interface;
    using lie_r = lie_interface;

    using tensor_basis_r = algebra_basis;
    using lie_basis_r = algebra_basis;

    virtual ~context() = default;

    // Get information about the context
    virtual deg_t width() const noexcept = 0;
    virtual deg_t depth() const noexcept = 0;
    const scalars::scalar_type* ctype() const noexcept { return p_ctype; }

    // Get related contexts
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth) const = 0;
    virtual std::shared_ptr<const context> get_alike(const scalars::scalar_type* new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth, const scalars::scalar_type* new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, const scalars::scalar_type* new_coeff) const = 0;

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
protected:
    void lie_to_tensor_fallback(free_tensor& result, const lie& arg) const;
    void tensor_to_lie_fallback(lie& result, const free_tensor& arg) const;

public:
    virtual free_tensor lie_to_tensor(const lie &arg) const = 0;
    virtual lie tensor_to_lie(const free_tensor &arg) const = 0;

    // Campbell-Baker-Hausdorff formula
protected:
    void cbh_fallback(free_tensor& collector, const std::vector<lie>& lies) const;

public:
    virtual lie cbh(const std::vector<lie> &lies, vector_type vtype) const = 0;


    // Methods for computing signatures
    virtual free_tensor to_signature(const lie &log_sig) const;
    virtual free_tensor signature(const signature_data& data) const = 0;
    virtual lie log_signature(const signature_data& data) const = 0;

    // Method for computing the derivative of a signature
    virtual free_tensor sig_derivative(
            const std::vector<derivative_compute_info> &info,
            vector_type vtype,
            vector_type) const = 0;
};

ESIG_ALGEBRA_EXPORT
std::shared_ptr<const context> get_context(
        deg_t width,
        deg_t depth,
        const scalars::scalar_type* ctype,
        const std::vector<std::string> &preferences = {});


struct ESIG_ALGEBRA_EXPORT context_maker {
    virtual ~context_maker() = default;
    virtual bool can_get(deg_t, deg_t, const scalars::scalar_type*) const noexcept = 0;
    virtual int get_priority(const std::vector<std::string> &preferences) const noexcept;
    virtual std::shared_ptr<const context> get_context(deg_t, deg_t, const scalars::scalar_type*) const = 0;
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


}// namespace algebra
}// namespace esig


#endif//ESIG_ALGEBRA_CONTEXT_H_
