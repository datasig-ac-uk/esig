//
// Created by user on 04/04/2022.
//

#include <esig/algebra/free_tensor_interface.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/algebra_traits.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>


using namespace esig;
using namespace esig::algebra;

class MockFreeTensor : public free_tensor_interface
{
public:
    using interface_t = free_tensor_interface;

//    MOCK_METHOD(algebra_iterator, begin, (), (const, override));
//    MOCK_METHOD(algebra_iterator, end, (), (const, override));
    MOCK_METHOD(dimn_t, size, (), (const, override));
    MOCK_METHOD(deg_t, degree, (), (const, override));
    MOCK_METHOD(deg_t, width, (), (const, override));
    MOCK_METHOD(deg_t, depth, (), (const, override));
    MOCK_METHOD(vector_type, storage_type, (), (const, noexcept, override));
    MOCK_METHOD(coefficient_type, coeff_type, (), (const, noexcept, override));

    MOCK_METHOD(coefficient, get, (key_type), (const, override));
    MOCK_METHOD(coefficient, get_mut, (key_type), (override));

    MOCK_METHOD(dense_data_access_iterator, iterate_dense_components, (), (const, override));

    MOCK_METHOD(algebra_iterator, begin, (), (const, override));
    MOCK_METHOD(algebra_iterator, end, (), (const, override));

    MOCK_METHOD(free_tensor, uminus, (), (const, override));
    MOCK_METHOD(free_tensor, add, (const free_tensor_interface &), (const, override));
    MOCK_METHOD(free_tensor, sub, (const free_tensor_interface &), (const, override));
    MOCK_METHOD(free_tensor, smul, (const coefficient &), (const, override));
    MOCK_METHOD(free_tensor, sdiv, (const coefficient &), (const, override));
    MOCK_METHOD(free_tensor, mul, (const free_tensor_interface &), (const, override));

    MOCK_METHOD(free_tensor_interface &, add_inplace, (const free_tensor_interface &), (override));
    MOCK_METHOD(free_tensor_interface &, sub_inplace, (const free_tensor_interface &), (override));
    MOCK_METHOD(free_tensor_interface &, smul_inplace, (const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, sdiv_inplace, (const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, mul_inplace, (const free_tensor_interface &), (override));

    MOCK_METHOD(free_tensor_interface &, add_scal_mul, (const free_tensor_interface &, const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, sub_scal_mul, (const free_tensor_interface &, const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, add_scal_div, (const free_tensor_interface &, const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, sub_scal_div, (const free_tensor_interface &, const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, add_mul, (const free_tensor_interface &, const free_tensor_interface &), (override));
    MOCK_METHOD(free_tensor_interface &, sub_mul, (const free_tensor_interface &, const free_tensor_interface &), (override));
    MOCK_METHOD(free_tensor_interface &, mul_smul, (const free_tensor_interface &, const coefficient &), (override));
    MOCK_METHOD(free_tensor_interface &, mul_sdiv, (const free_tensor_interface &, const coefficient &), (override));

    MOCK_METHOD(free_tensor, exp, (), (const, override));
    MOCK_METHOD(free_tensor, log, (), (const, override));
    MOCK_METHOD(free_tensor, inverse, (), (const, override));
    MOCK_METHOD(free_tensor_interface&, fmexp, (const free_tensor_interface&), (override));

    MOCK_METHOD(std::ostream &, print, (std::ostream &), (const, override));

    MOCK_METHOD(bool, equals, (const free_tensor_interface &), (const, override));
};

using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::_;

class FreeTensorWrapperFixture : public ::testing::Test
{
protected:
    free_tensor tensorobj;
    free_tensor othertensor;


    static MockFreeTensor &mocked(free_tensor &arg) noexcept
    {
        return dynamic_cast<MockFreeTensor&>(*algebra_base_access::get(arg));
    }

    static free_tensor setup_mock_tensor()
    {
        free_tensor result = free_tensor::from_args<MockFreeTensor>();
        auto &mtensor = mocked(result);

        ON_CALL(mtensor, size()).WillByDefault(Return(0));
        ON_CALL(mtensor, degree()).WillByDefault(Return(0));
        ON_CALL(mtensor, width()).WillByDefault(Return(5));
        ON_CALL(mtensor, depth()).WillByDefault(Return(2));
        ON_CALL(mtensor, storage_type()).WillByDefault(Return(vector_type::dense));
        ON_CALL(mtensor, coeff_type()).WillByDefault(Return(coefficient_type::dp_real));
        ON_CALL(mtensor, uminus()).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, add(_)).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, sub(_)).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, smul(_)).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, sdiv(_)).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, mul(_)).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, add_inplace(_)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, sub_inplace(_)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, smul_inplace(_)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, sdiv_inplace(_)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, mul_inplace(_)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, add_scal_mul(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, sub_scal_mul(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, add_scal_div(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, sub_scal_div(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, add_mul(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, sub_mul(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, mul_smul(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, mul_sdiv(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, print).WillByDefault(testing::ReturnArg<0>());
        ON_CALL(mtensor, equals(_)).WillByDefault(Return(false));
        ON_CALL(mtensor, exp()).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, log()).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, inverse()).WillByDefault(Return(free_tensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, fmexp(_)).WillByDefault(ReturnRef(mtensor));
        return result;
    }


    FreeTensorWrapperFixture() : tensorobj(setup_mock_tensor()), othertensor(setup_mock_tensor())
    {
    }
};


TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSize)
{
    EXPECT_CALL(mocked(tensorobj), size()).Times(1);

    auto s = tensorobj.size();
    ASSERT_EQ(s, 0);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceDegree)
{
    EXPECT_CALL(mocked(tensorobj), degree()).Times(1);

    auto s = tensorobj.degree();
    ASSERT_EQ(s, 0);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceWidth)
{
    EXPECT_CALL(mocked(tensorobj), width()).Times(1);

    auto s = tensorobj.width();
    ASSERT_EQ(s, 5);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceDepth)
{
    EXPECT_CALL(mocked(tensorobj), depth()).Times(1);

    auto s = tensorobj.depth();
    ASSERT_EQ(s, 2);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceUminus)
{
    EXPECT_CALL(mocked(tensorobj), uminus()).Times(1);

    auto v = tensorobj.uminus();
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAdd)
{
    EXPECT_CALL(mocked(tensorobj), add(_)).Times(1);

    auto v = tensorobj.add(othertensor);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSub)
{
    EXPECT_CALL(mocked(tensorobj), sub(_)).Times(1);

    auto v = tensorobj.sub(othertensor);
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceMul)
{
    EXPECT_CALL(mocked(tensorobj), mul(_)).Times(1);

    auto v = tensorobj.mul(othertensor);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSmul)
{
    EXPECT_CALL(mocked(tensorobj), smul(_)).Times(1);
    auto v = tensorobj.smul(coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSdiv)
{
    EXPECT_CALL(mocked(tensorobj), sdiv(_)).Times(1);
    auto v = tensorobj.sdiv(coefficient(1.0));
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddInplace)
{
    EXPECT_CALL(mocked(tensorobj), add_inplace(_)).Times(1);
    auto v = tensorobj.add_inplace(othertensor);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubInplace)
{
    EXPECT_CALL(mocked(tensorobj), sub_inplace(_)).Times(1);
    auto v = tensorobj.sub_inplace(othertensor);
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceMulInplace)
{
    EXPECT_CALL(mocked(tensorobj), mul_inplace(_)).Times(1);
    auto v = tensorobj.mul_inplace(othertensor);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSMulInplace)
{
    EXPECT_CALL(mocked(tensorobj), smul_inplace(_)).Times(1);
    auto v = tensorobj.smul_inplace(coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSDivInplace)
{
    EXPECT_CALL(mocked(tensorobj), sdiv_inplace(_)).Times(1);
    auto v = tensorobj.sdiv_inplace(coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddScalMul)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_mul(_, _)).Times(1);
    auto v = tensorobj.add_scal_mul(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubScalMul)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_mul(_, _)).Times(1);
    auto v = tensorobj.add_scal_mul(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddScalDiv)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_div(_, _)).Times(1);
    auto v = tensorobj.add_scal_div(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubScalDiv)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_div(_, _)).Times(1);
    auto v = tensorobj.add_scal_div(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddMul)
{
    EXPECT_CALL(mocked(tensorobj), add_mul(_, _)).Times(1);
    auto other = setup_mock_tensor();
    auto v = tensorobj.add_mul(othertensor, other);
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubMul)
{
    EXPECT_CALL(mocked(tensorobj), sub_mul(_, _)).Times(1);
    auto other = setup_mock_tensor();
    auto v = tensorobj.sub_mul(othertensor, other);
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceMulSMul)
{
    EXPECT_CALL(mocked(tensorobj), mul_smul(_, _)).Times(1);
    auto v = tensorobj.mul_smul(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceMulSDiv)
{
    EXPECT_CALL(mocked(tensorobj), mul_sdiv(_, _)).Times(1);
    auto v = tensorobj.mul_sdiv(othertensor, coefficient(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfacePrint)
{
    EXPECT_CALL(mocked(tensorobj), print(_)).Times(1);
    std::stringstream ss;
    tensorobj.print(ss);
}

TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceEqualsTrue)
{
    EXPECT_CALL(mocked(tensorobj), equals(_))
            .Times(1)
            .WillOnce(Return(true));
    ASSERT_EQ(tensorobj, othertensor);
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceEqualsFalse)
{
    EXPECT_CALL(mocked(tensorobj), equals(_))
            .Times(1)
            .WillOnce(Return(false));
    auto result = tensorobj != othertensor;
}
