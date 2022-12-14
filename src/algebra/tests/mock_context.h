//
// Created by user on 05/04/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TESTS_MOCK_CONTEXT_H_
#define ESIG_PATHS_SRC_ALGEBRA_TESTS_MOCK_CONTEXT_H_

#include <esig/algebra/context.h>
#include <gmock/gmock.h>

namespace esig {
namespace testing {

class MockContext : public algebra::context
{
public:
    MOCK_METHOD(deg_t, width, (), (const, noexcept, override));
    MOCK_METHOD(deg_t, depth, (), (const, noexcept, override));
    MOCK_METHOD(dimn_t, lie_size, (deg_t), (const, noexcept, override));
    MOCK_METHOD(dimn_t, tensor_size, (deg_t), (const, noexcept, override));

    MOCK_METHOD(algebra::free_tensor, construct_tensor, (const algebra::vector_construction_data&), (const, override));
    MOCK_METHOD(algebra::lie, construct_lie, (const algebra::vector_construction_data&), (const, override));
    MOCK_METHOD(algebra::free_tensor, zero_tensor, (algebra::vector_type), (const, override));
    MOCK_METHOD(algebra::lie, zero_lie, (algebra::vector_type), (const, override));

    using lie_vec_t = std::vector<algebra::lie>;
    MOCK_METHOD(algebra::lie, cbh, (const lie_vec_t&, algebra::vector_type), (const, override));

    MOCK_METHOD(algebra::free_tensor, to_signature, (const algebra::lie&), (const, override));
    MOCK_METHOD(algebra::free_tensor, signature, (const algebra::signature_data&), (const, override));
    MOCK_METHOD(algebra::lie, log_signature, (const algebra::signature_data&), (const, override));

    using vec_deriv_info = std::vector<algebra::derivative_compute_info>;
    MOCK_METHOD(algebra::free_tensor, sig_derivative, (const vec_deriv_info&, algebra::vector_type, algebra::vector_type), (const, override));

};

} // namespace testing
} // namespace esig



#endif//ESIG_PATHS_SRC_ALGEBRA_TESTS_MOCK_CONTEXT_H_
