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

    MockContext() : algebra::context(nullptr) {}

    MOCK_METHOD(deg_t, width, (), (const, noexcept, override));
    MOCK_METHOD(deg_t, depth, (), (const, noexcept, override));
    MOCK_METHOD(dimn_t, lie_size, (deg_t), (const, noexcept, override));
    MOCK_METHOD(dimn_t, tensor_size, (deg_t), (const, noexcept, override));

    MOCK_METHOD(algebra::FreeTensor, construct_tensor, (const algebra::vector_construction_data&), (const, override));
    MOCK_METHOD(algebra::Lie, construct_lie, (const algebra::vector_construction_data&), (const, override));
    MOCK_METHOD(algebra::FreeTensor, zero_tensor, (algebra::VectorType), (const, override));
    MOCK_METHOD(algebra::Lie, zero_lie, (algebra::VectorType), (const, override));

    using lie_vec_t = std::vector<algebra::Lie>;
    MOCK_METHOD(algebra::Lie, cbh, (const lie_vec_t&, algebra::VectorType), (const, override));

    MOCK_METHOD(algebra::FreeTensor, to_signature, (const algebra::Lie&), (const, override));
    MOCK_METHOD(algebra::FreeTensor, signature, (const algebra::signature_data&), (const, override));
    MOCK_METHOD(algebra::Lie, log_signature, (const algebra::signature_data&), (const, override));

    using vec_deriv_info = std::vector<algebra::derivative_compute_info>;
    MOCK_METHOD(algebra::FreeTensor, sig_derivative, (const vec_deriv_info&, algebra::VectorType, algebra::VectorType), (const, override));

};

} // namespace testing
} // namespace esig



#endif//ESIG_PATHS_SRC_ALGEBRA_TESTS_MOCK_CONTEXT_H_
