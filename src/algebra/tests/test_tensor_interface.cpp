//
// Created by user on 04/04/2022.
//

#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/tensor_interface.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace esig;
using namespace esig::algebra;

class MockFreeTensor : public FreeTensorInterface
{
public:
    using interface_t = FreeTensorInterface;
    using algebra_interface_t = AlgebraInterface<FreeTensor>;

//    MOCK_METHOD(algebra_iterator, begin, (), (const, override));
//    MOCK_METHOD(algebra_iterator, end, (), (const, override));
    MOCK_METHOD(dimn_t, size, (), (const, override));
    MOCK_METHOD(deg_t, degree, (), (const, override));
    MOCK_METHOD(deg_t, width, (), (const, override));
    MOCK_METHOD(deg_t, depth, (), (const, override));
    MOCK_METHOD(VectorType, storage_type, (), (const, noexcept, override));
    MOCK_METHOD(const scalars::ScalarType *, coeff_type, (), (const, noexcept, override));

    MOCK_METHOD(scalars::Scalar, get, (key_type), (const, override));
    MOCK_METHOD(scalars::Scalar, get_mut, (key_type), (override));

//    MOCK_METHOD(dense_data_access_iterator, iterate_dense_components, (), (const, override));

    MOCK_METHOD(algebra_iterator, begin, (), (const, override));
    MOCK_METHOD(algebra_iterator, end, (), (const, override));

    MOCK_METHOD(FreeTensor, uminus, (), (const, override));
    MOCK_METHOD(FreeTensor, add, (const algebra_interface_t &), (const, override));
    MOCK_METHOD(FreeTensor, sub, (const algebra_interface_t &), (const, override));
    MOCK_METHOD(FreeTensor, smul, (const scalars::Scalar&), (const, override));
    MOCK_METHOD(FreeTensor, sdiv, (const scalars::Scalar&), (const, override));
    MOCK_METHOD(FreeTensor, mul, (const algebra_interface_t &), (const, override));

    MOCK_METHOD(void, add_inplace, (const algebra_interface_t &), (override));
    MOCK_METHOD(void, sub_inplace, (const algebra_interface_t &), (override));
    MOCK_METHOD(void, smul_inplace, (const scalars::Scalar&), (override));
    MOCK_METHOD(void, sdiv_inplace, (const scalars::Scalar&), (override));
    MOCK_METHOD(void, mul_inplace, (const algebra_interface_t &), (override));

    MOCK_METHOD(void, add_scal_mul, (const algebra_interface_t &, const scalars::Scalar&), (override));
    MOCK_METHOD(void, sub_scal_mul, (const algebra_interface_t &, const scalars::Scalar&), (override));
    MOCK_METHOD(void, add_scal_div, (const algebra_interface_t &, const scalars::Scalar&), (override));
    MOCK_METHOD(void, sub_scal_div, (const algebra_interface_t &, const scalars::Scalar&), (override));
    MOCK_METHOD(void, add_mul, (const algebra_interface_t &, const algebra_interface_t &), (override));
    MOCK_METHOD(void, sub_mul, (const algebra_interface_t &, const algebra_interface_t &), (override));
    MOCK_METHOD(void, mul_smul, (const algebra_interface_t &, const scalars::Scalar&), (override));
    MOCK_METHOD(void, mul_sdiv, (const algebra_interface_t &, const scalars::Scalar&), (override));

    MOCK_METHOD(FreeTensor, exp, (), (const, override));
    MOCK_METHOD(FreeTensor, log, (), (const, override));
    MOCK_METHOD(FreeTensor, inverse, (), (const, override));
    MOCK_METHOD(FreeTensor, antipode, (), (const, override));
    MOCK_METHOD(void, fmexp, (const FreeTensor&), (override));

    MOCK_METHOD(std::ostream &, print, (std::ostream &), (const, override));

    MOCK_METHOD(bool, equals, (const algebra_interface_t &), (const, override));
};

using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::_;

class FreeTensorWrapperFixture : public ::testing::Test
{
protected:
    FreeTensor tensorobj;
    FreeTensor othertensor;


    static MockFreeTensor &mocked(FreeTensor &arg) noexcept
    {
        return static_cast<MockFreeTensor&>(*arg);
    }

    static FreeTensor  setup_mock_tensor()
    {
        FreeTensor result = FreeTensor::from_args<MockFreeTensor>();
        auto &mtensor = mocked(result);

        ON_CALL(mtensor, size()).WillByDefault(Return(0));
        ON_CALL(mtensor, degree()).WillByDefault(Return(0));
        ON_CALL(mtensor, width()).WillByDefault(Return(5));
        ON_CALL(mtensor, depth()).WillByDefault(Return(2));
        ON_CALL(mtensor, storage_type()).WillByDefault(Return(VectorType::dense));
        ON_CALL(mtensor, coeff_type()).WillByDefault(Return(scalars::dtl::scalar_type_trait<double>::get_type()));
        ON_CALL(mtensor, uminus()).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, add(_)).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, sub(_)).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, smul(_)).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, sdiv(_)).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, mul(_)).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
//        ON_CALL(mtensor, add_inplace(_)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, sub_inplace(_)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, smul_inplace(_)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, sdiv_inplace(_)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, mul_inplace(_)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, add_scal_mul(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, sub_scal_mul(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, add_scal_div(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, sub_scal_div(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, add_mul(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, sub_mul(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, mul_smul(_, _)).WillByDefault(ReturnRef(mtensor));
//        ON_CALL(mtensor, mul_sdiv(_, _)).WillByDefault(ReturnRef(mtensor));
        ON_CALL(mtensor, print).WillByDefault(testing::ReturnArg<0>());
        ON_CALL(mtensor, equals(_)).WillByDefault(Return(false));
        ON_CALL(mtensor, exp()).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, log()).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, inverse()).WillByDefault(Return(FreeTensor::from_args<MockFreeTensor>()));
        ON_CALL(mtensor, fmexp(_)).WillByDefault(Return());
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
    auto v = tensorobj.smul(scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSdiv)
{
    EXPECT_CALL(mocked(tensorobj), sdiv(_)).Times(1);
    auto v = tensorobj.sdiv(scalars::Scalar(1.0));
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
    auto v = tensorobj.smul_inplace(scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSDivInplace)
{
    EXPECT_CALL(mocked(tensorobj), sdiv_inplace(_)).Times(1);
    auto v = tensorobj.sdiv_inplace(scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddScalMul)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_mul(_, _)).Times(1);
    auto v = tensorobj.add_scal_mul(othertensor, scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubScalMul)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_mul(_, _)).Times(1);
    auto v = tensorobj.add_scal_mul(othertensor, scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceAddScalDiv)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_div(_, _)).Times(1);
    auto v = tensorobj.add_scal_div(othertensor, scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceSubScalDiv)
{
    EXPECT_CALL(mocked(tensorobj), add_scal_div(_, _)).Times(1);
    auto v = tensorobj.add_scal_div(othertensor, scalars::Scalar(1.0));
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
    auto v = tensorobj.mul_smul(othertensor, scalars::Scalar(1.0));
}
TEST_F(FreeTensorWrapperFixture, TestFreeTensorInterfaceMulSDiv)
{
    EXPECT_CALL(mocked(tensorobj), mul_sdiv(_, _)).Times(1);
    auto v = tensorobj.mul_sdiv(othertensor, scalars::Scalar(1.0));
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
