//
// Created by user on 20/03/2022.
//

#ifndef ESIG_ALGEBRA_CONTEXT_H_
#define ESIG_ALGEBRA_CONTEXT_H_

#include <esig/algebra/context_fwd.h>

#include <esig/scalars.h>

#include "algebra_fwd.h"
#include "algebra_base.h"
#include "basis.h"
#include "lie.h"
#include "free_tensor.h"
#include "shuffle_tensor.h"

namespace esig {
namespace algebra {



struct signature_data
{
    scalars::ScalarStream data_stream;
    std::vector<const key_type*> key_stream;
    VectorType vect_type;
};


struct derivative_compute_info {
    Lie logsig_of_interval;
    Lie perturbation;
};


struct vector_construction_data
{
    scalars::KeyScalarArray data;
    VectorType vect_type = VectorType::sparse;
};


/**
 * @brief Context object that controls creation and manipulation of algebra types.
 *
 *
 */
class ESIG_ALGEBRA_EXPORT context
{
    const scalars::ScalarType * p_ctype;

protected:

    explicit context(const scalars::ScalarType * ctype) : p_ctype(ctype)
    {}

public:
    using tensor_r = FreeTensorInterface;
    using lie_r = LieInterface;

    using tensor_basis_r = Basis;
    using lie_basis_r = Basis;

    virtual ~context() = default;

    // Get information about the context
    virtual deg_t width() const noexcept = 0;
    virtual deg_t depth() const noexcept = 0;
    const scalars::ScalarType * ctype() const noexcept { return p_ctype; }

    // Get related contexts
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth) const = 0;
    virtual std::shared_ptr<const context> get_alike(const scalars::ScalarType * new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_depth, const scalars::ScalarType * new_coeff) const = 0;
    virtual std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, const scalars::ScalarType * new_coeff) const = 0;

    // Get information about the tensor and Lie types
    virtual dimn_t lie_size(deg_t d) const noexcept = 0;
    virtual dimn_t tensor_size(deg_t d) const noexcept = 0;

    virtual bool check_compatible(const context& other) const noexcept;

    virtual FreeTensor convert(const FreeTensor& arg, VectorType new_vec_type) const = 0;
    virtual Lie convert(const Lie& arg, VectorType new_vec_type) const = 0;

    //TODO: needs more thought

    // Access the basis interface for Lie and tensors
    //virtual std::shared_ptr<algebra_basis> get_tensor_basis() const noexcept = 0;
    //virtual std::shared_ptr<algebra_basis> get_Lie_basis() const noexcept = 0;
//    virtual const algebra_basis &borrow_tbasis() const noexcept = 0;
//    virtual const algebra_basis &borrow_lbasis() const noexcept = 0;

    virtual Basis get_tensor_basis() const = 0;
    virtual Basis get_lie_basis() const = 0;


    // Construct new instances of tensors and Lies
    virtual FreeTensor construct_tensor(const vector_construction_data &) const = 0;
    virtual Lie construct_lie(const vector_construction_data &) const = 0;
    virtual FreeTensor zero_tensor(VectorType vtype) const;
    virtual Lie zero_lie(VectorType vtype) const;

    // Conversions between Lie and Tensors
protected:
    void lie_to_tensor_fallback(FreeTensor& result, const Lie& arg) const;
    void tensor_to_lie_fallback(Lie& result, const FreeTensor& arg) const;

public:
    virtual FreeTensor lie_to_tensor(const Lie &arg) const = 0;
    virtual Lie tensor_to_lie(const FreeTensor &arg) const = 0;

    // Campbell-Baker-Hausdorff formula
protected:
    void cbh_fallback(FreeTensor& collector, const std::vector<Lie>& Lies) const;

public:
    virtual Lie cbh(const std::vector<Lie> &Lies, VectorType vtype) const = 0;


    // Methods for computing signatures
    virtual FreeTensor to_signature(const Lie &log_sig) const;
    virtual FreeTensor signature(const signature_data& data) const = 0;
    virtual Lie log_signature(const signature_data& data) const = 0;

    // Method for computing the derivative of a signature
    virtual FreeTensor sig_derivative(
            const std::vector<derivative_compute_info> &info,
        VectorType vtype,
        VectorType) const = 0;
};

ESIG_ALGEBRA_EXPORT
std::shared_ptr<const context> get_context(
        deg_t width,
        deg_t depth,
        const scalars::ScalarType * ctype,
        const std::vector<std::string> &preferences = {});


struct ESIG_ALGEBRA_EXPORT context_maker {
    virtual ~context_maker() = default;
    virtual bool can_get(deg_t, deg_t, const scalars::ScalarType *) const noexcept = 0;
    virtual int get_priority(const std::vector<std::string> &preferences) const noexcept;
    virtual std::shared_ptr<const context> get_context(deg_t, deg_t, const scalars::ScalarType *) const = 0;
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
