//
// Created by sam on 30/03/2022.
//
/**
 * This header file defines various types of path and associated metadata structs.
 *
 * Abstractly, a path is an object that can be queried to produce a log-signature
 * (or signature) on a particular interval. This is described by the `path_interface`
 * class. Each implementation of a path derives from this base class and is then
 * wrapped - using type erasure - in a `path` object.
 *
 * Path implementations do not compute log signatures by themselves. Instead, they
 * pass the data to a context object that can produce the desired output type.
 * The context is usually the same as was used to create the path or can be provided
 * by the user.
 *
 * Also provided here are several convenience subclasses such as the
 * `dyadic_caching_layer`, which caches intermediate signature calculations
 * over a dyadic dissection of the domain to help improve computation times.
 * There is also a subclass of `path_interface` representing paths that arise
 * as the solution to controlled differential equations, which has an additional
 * `base_point` method that returns the initial value of the path.
 */
#ifndef ESIG_PATHS_PATH_H_
#define ESIG_PATHS_PATH_H_

#include <esig/implementation_types.h>
#include <esig/intervals.h>
#include <esig/paths/esig_paths_export.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/algebra_traits.h>

#include <esig/algebra/free_tensor_interface.h>
#include <esig/algebra/lie_interface.h>
#include <esig/algebra/context.h>

#include <boost/container/map.hpp>

#include <memory>
#include <mutex>
#include <utility>

namespace esig {
namespace paths {

/**
 * @brief Metadata associated with all path objects
 *
 * This struct holds various pieces of data about the space in which a path
 * lies: the underlying vector space dimension; the type of coefficients; the
 * effective domain of the path (where the values of the path are concentrated);
 * the truncation depth for signatures and log-signatures; how the data is stored
 * in the path; and the storage model for the free tensor signatures and Lie log-
 * signatures.
 */
struct path_metadata {
    deg_t width;
    deg_t depth;
    real_interval effective_domain;
    std::shared_ptr<algebra::context> ctx;

    algebra::coefficient_type ctype;
    algebra::input_data_type data_type;
    algebra::vector_type result_vec_type;
};

/**
 * @brief Base class for all path types.
 *
 * An abstract path provides methods for querying the signature or
 * log-signature over an interval in the parameter space, returning either
 * a free tensor or Lie element. This base class has establishes this interface
 * and also acts as a holder for the path metadata.
 *
 * Path implementations should implement the `log_signature` virtual function
 * (taking `interval` and `context` arguments) that is used to implement the
 * other flavours of computation methods. (Note that signatures a computed
 * from log signatures, rather than using the data to compute these
 * independently.)
 */
class ESIG_PATHS_EXPORT path_interface
{
    path_metadata m_metadata;

public:
    using accuracy_t = param_t;
    using compute_depth_t = unsigned;

    explicit path_interface(path_metadata metadata);

    virtual ~path_interface() = default;

    virtual const path_metadata &metadata() const noexcept;
    virtual compute_depth_t compute_depth(accuracy_t accuracy) const noexcept;
    virtual bool empty(const interval &domain) const;

    virtual algebra::lie
    log_signature(const interval &domain, const algebra::context &ctx) const = 0;

    virtual algebra::lie
    log_signature(const dyadic_interval &domain, compute_depth_t resolution, const algebra::context &ctx) const;

    virtual algebra::lie
    log_signature(const interval &domain, compute_depth_t resolution, const algebra::context &ctx) const;

    virtual algebra::free_tensor
    signature(const interval &domain, compute_depth_t resolution, const algebra::context &ctx) const;
};

/**
 * @brief Subclass of `path_interface` for solutions of controlled differential equations.
 */
class ESIG_PATHS_EXPORT solution_path_interface : public path_interface
{
public:
    using path_interface::path_interface;
    virtual algebra::lie
    base_point() const = 0;
};

/**
 * @brief Caching layer utilising a dyadic dissection of the parameter interval.
 *
 * This layer introducing caching for the computation of log signatures by
 * utilising the fact that the signature of a concatenation of paths is the
 * product of signatures (or applying the Campbell-Baker-Hausdorff formula to
 * log signatures). The parameter interval is dissected into dyadic intervals
 * of a resolution and the log signature is computed on all those dyadic intervals
 * that are contained within the requested interval. These are then combined
 * using the Campbell-Baker-Hausdorff formula to give the log signature over the
 * whole interval.
 *
 */
class ESIG_PATHS_EXPORT dyadic_caching_layer : public path_interface
{
    mutable boost::container::map<dyadic_interval, algebra::lie> m_cache;
    mutable std::recursive_mutex m_compute_lock;

public:
    using path_interface::log_signature;

    using path_interface::path_interface;
    dyadic_caching_layer(const dyadic_caching_layer&) = delete;
    dyadic_caching_layer(dyadic_caching_layer&&) noexcept;

    dyadic_caching_layer& operator=(const dyadic_caching_layer&) = delete;
    dyadic_caching_layer& operator=(dyadic_caching_layer&&) noexcept;


    algebra::lie
    log_signature(const dyadic_interval &domain, compute_depth_t resolution, const algebra::context &ctx) const override;
};

/**
 * @brief Path that is constructed dynamically over intervals.
 *
 * This class is a base for path types that are generated dynamically over a
 * requested interval and don't hold any inherent "data" of their own. For example,
 * if a path is defined by an explicit formula then this is used to compute
 * the Lie increments used to compute the log signature over each interval lazily.
 * This allows us to query the log signature to greater accuracy where this matters
 * or less accuracy when speed is important.
 *
 * This class makes use of the dyadic caching layer to speed up computations,
 * and to prevent unnecessary calls to the generating function. If a greater
 * dyadic resolution is requested, old cached values are evicted in favour of
 * the combination of log signatures computed with higher resolution.
 */
struct ESIG_PATHS_EXPORT dynamically_constructed_path : dyadic_caching_layer
{
    using dyadic_caching_layer::dyadic_caching_layer;

    virtual algebra::allocating_data_buffer eval(const interval &domain) const = 0;

    bool empty(const interval &domain) const override;

    using dyadic_caching_layer::log_signature;
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

/**
 * @brief Type erased container and main interface for path objects.
 *
 *
 */
class ESIG_PATHS_EXPORT path
{
    std::unique_ptr<const path_interface> p_impl;
    std::shared_ptr<algebra::context> get_default_context() const;

    friend struct path_base_access;

public:
    using accuracy_t = path_interface::accuracy_t;
    using perturbation_t = std::pair<real_interval, algebra::lie>;
    using perturbation_list_t = std::vector<perturbation_t>;

    path() = default;
    path(path&& other) noexcept = default;


    template<typename Impl>
    explicit path(Impl &&p);

    const path_metadata& metadata() const noexcept;

    algebra::lie log_signature(
            const interval &domain,
            accuracy_t accuracy) const;

    algebra::lie log_signature(
            const interval &domain,
            accuracy_t accuracy,
            const algebra::context &ctx) const;

    algebra::lie log_signature(accuracy_t accuracy) const;
    algebra::lie log_signature(accuracy_t accuracy, const algebra::context& ctx) const;

    algebra::free_tensor signature(
            const interval &domain,
            accuracy_t accuracy) const;
    algebra::free_tensor signature(
            const interval &domain,
            accuracy_t accuracy,
            const algebra::context &ctx) const;

    algebra::free_tensor signature(accuracy_t accuracy) const;
    algebra::free_tensor signature(accuracy_t accuracy, const algebra::context& ctx) const;

    algebra::free_tensor
    signature_derivative(const interval &domain,
                         const algebra::lie &perturbation,
                         accuracy_t accuracy) const;
    algebra::free_tensor
    signature_derivative(const interval &domain,
                         const algebra::lie &perturbation,
                         accuracy_t accuracy,
                         const algebra::context &ctx) const;
    algebra::free_tensor
    signature_derivative(const perturbation_list_t &perturbations,
                         accuracy_t accuracy) const;
    algebra::free_tensor
    signature_derivative(const perturbation_list_t &perturbations,
                         accuracy_t accuracy,
                         const algebra::context &ctx) const;
};







struct ESIG_PATHS_EXPORT path_base_access
{
    static const path_interface& get(const path& p) noexcept;
};

// Implementation of the templated constructor of paths
template<typename Impl>
path::path(Impl &&p)
    : p_impl(new Impl(std::forward<Impl>(p)))
{
}

} // namespace paths
} // namespace esig


#endif//ESIG_PATHS_PATH_H_
