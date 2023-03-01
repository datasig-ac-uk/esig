//
// Created by user on 04/04/2022.
//

#include <esig/algebra/lie_interface.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <utility>
#include <sstream>

using namespace esig;
using namespace esig::algebra;

class MockLie;

namespace esig {
namespace algebra {
namespace dtl {

template <>
struct implementation_wrapper_selection<lie_interface, MockLie>
{
    using type = MockLie;
};
}
}
}

class MockLie : public esig::algebra::lie_interface
{
public:
    using scalar_type = double;
    using interface_t = esig::algebra::lie_interface;

    MOCK_METHOD(algebra_iterator, begin, (), (const, override));
    MOCK_METHOD(algebra_iterator, end, (), (const, override));
    MOCK_METHOD(dimn_t, size, (), (const, override));
    MOCK_METHOD(deg_t, degree, (), (const, override));
    MOCK_METHOD(deg_t, width, (), (const, override));
    MOCK_METHOD(deg_t, depth, (), (const, override));
    MOCK_METHOD(vector_type, storage_type, (), (const, noexcept, override));
    MOCK_METHOD(const scalars::ScalarType *, coeff_type, (), (const, noexcept, override));

//    MOCK_METHOD(dense_data_access_iterator, iterate_dense_components, (), (const, override));
    MOCK_METHOD(scalars::Scalar, get, (key_type), (const, override));
    MOCK_METHOD(scalars::Scalar, get_mut, (key_type), (override));

    MOCK_METHOD(lie, uminus, (), (const, override));
    MOCK_METHOD(lie, add, (const lie_interface&), (const, override));
    MOCK_METHOD(lie, sub, (const lie_interface&), (const, override));
    MOCK_METHOD(lie, smul, (const scalars::Scalar&), (const, override));
    MOCK_METHOD(lie, sdiv, (const scalars::Scalar&), (const, override));
    MOCK_METHOD(lie, mul, (const lie_interface&), (const, override));

    MOCK_METHOD(void, add_inplace, (const lie_interface&), (override));
    MOCK_METHOD(void, sub_inplace, (const lie_interface&), (override));
    MOCK_METHOD(void, smul_inplace, (const scalars::Scalar&), (override));
    MOCK_METHOD(void, sdiv_inplace, (const scalars::Scalar&), (override));
    MOCK_METHOD(void, mul_inplace, (const lie_interface&), (override));

    MOCK_METHOD(void, add_scal_mul, (const lie_interface&, const scalars::Scalar&), (override));
    MOCK_METHOD(void, sub_scal_mul, (const lie_interface&, const scalars::Scalar&), (override));
    MOCK_METHOD(void, add_scal_div, (const lie_interface&, const scalars::Scalar&), (override));
    MOCK_METHOD(void, sub_scal_div, (const lie_interface&, const scalars::Scalar&), (override));
    MOCK_METHOD(void, add_mul, (const lie_interface&, const lie_interface&), (override));
    MOCK_METHOD(void, sub_mul, (const lie_interface&, const lie_interface&), (override));
    MOCK_METHOD(void, mul_smul, (const lie_interface&, const scalars::Scalar&), (override));
    MOCK_METHOD(void, mul_sdiv, (const lie_interface&, const scalars::Scalar&), (override));

    MOCK_METHOD(std::ostream&, print, (std::ostream&), (const, override));

    MOCK_METHOD(bool, equals, (const lie_interface&), (const, override));
};

using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::_;

class LieWrapperFixture : public ::testing::Test
{
protected:
    lie lieobj;
    lie otherlie;


    static MockLie &mocked(lie &arg) noexcept
    {
        return dynamic_cast<MockLie&>(*algebra_access<lie_interface>::get(arg));
    }

    static lie setup_mock_lie()
    {
        lie result = lie::from_args<MockLie>();
        auto& mlie = mocked(result);

        ON_CALL(mlie, size()).WillByDefault(Return(0));
        ON_CALL(mlie, degree()).WillByDefault(Return(0));
        ON_CALL(mlie, width()).WillByDefault(Return(5));
        ON_CALL(mlie, depth()).WillByDefault(Return(2));
        ON_CALL(mlie, storage_type()).WillByDefault(Return(vector_type::dense));
        ON_CALL(mlie, coeff_type()).WillByDefault(Return(scalars::dtl::scalar_type_trait<double>::get_type()));
        ON_CALL(mlie, uminus()).WillByDefault(Return(lie::from_args<MockLie>()));
        ON_CALL(mlie, add(_)).WillByDefault(Return(lie::from_args<MockLie>()));
        ON_CALL(mlie, sub(_)).WillByDefault(Return(lie::from_args<MockLie>()));
        ON_CALL(mlie, smul(_)).WillByDefault(Return(lie::from_args<MockLie>()));
        ON_CALL(mlie, sdiv(_)).WillByDefault(Return(lie::from_args<MockLie>()));
        ON_CALL(mlie, mul(_)).WillByDefault(Return(lie::from_args<MockLie>()));
//        ON_CALL(mlie, add_inplace(_)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, sub_inplace(_)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, smul_inplace(_)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, sdiv_inplace(_)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, mul_inplace(_)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, add_scal_mul(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, sub_scal_mul(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, add_scal_div(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, sub_scal_div(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, add_mul(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, sub_mul(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, mul_smul(_, _)).WillByDefault(ReturnRef(mlie));
//        ON_CALL(mlie, mul_sdiv(_, _)).WillByDefault(ReturnRef(mlie));
        ON_CALL(mlie, print(_)).WillByDefault(testing::ReturnArg<0>());
        ON_CALL(mlie, equals(_)).WillByDefault(Return(false));
        return result;
    }


    LieWrapperFixture() : lieobj(setup_mock_lie()), otherlie(setup_mock_lie())
    {
    }


};


TEST_F(LieWrapperFixture, TestLieInterfaceSize)
{
    EXPECT_CALL(mocked(lieobj), size()).Times(1);

    auto s = lieobj.size();
    ASSERT_EQ(s, 0);
}

TEST_F(LieWrapperFixture, TestLieInterfaceDegree)
{
    EXPECT_CALL(mocked(lieobj), degree()).Times(1);

    auto s = lieobj.degree();
    ASSERT_EQ(s, 0);
}

TEST_F(LieWrapperFixture, TestLieInterfaceWidth)
{
    EXPECT_CALL(mocked(lieobj), width()).Times(1);

    auto s = lieobj.width();
    ASSERT_EQ(s, 5);
}

TEST_F(LieWrapperFixture, TestLieInterfaceDepth)
{
    EXPECT_CALL(mocked(lieobj), depth()).Times(1);

    auto s = lieobj.depth();
    ASSERT_EQ(s, 2);
}

TEST_F(LieWrapperFixture, TestLieInterfaceUminus)
{
    EXPECT_CALL(mocked(lieobj), uminus()).Times(1);

    auto v = lieobj.uminus();
}

TEST_F(LieWrapperFixture, TestLieInterfaceAdd)
{
    EXPECT_CALL(mocked(lieobj), add(_)).Times(1);

    auto v = lieobj.add(otherlie);
}

TEST_F(LieWrapperFixture, TestLieInterfaceSub)
{
    EXPECT_CALL(mocked(lieobj), sub(_)).Times(1);

    auto v = lieobj.sub(otherlie);
}
TEST_F(LieWrapperFixture, TestLieInterfaceMul)
{
    EXPECT_CALL(mocked(lieobj), mul(_)).Times(1);

    auto v = lieobj.mul(otherlie);
}

TEST_F(LieWrapperFixture, TestLieInterfaceSmul)
{
    EXPECT_CALL(mocked(lieobj), smul(_)).Times(1);
    auto v = lieobj.smul(scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceSdiv)
{
    EXPECT_CALL(mocked(lieobj), sdiv(_)).Times(1);
    auto v = lieobj.sdiv(scalars::Scalar(1.0));
}

TEST_F(LieWrapperFixture, TestLieInterfaceAddInplace)
{
    EXPECT_CALL(mocked(lieobj), add_inplace(_)).Times(1);
    auto v = lieobj.add_inplace(otherlie);
}

TEST_F(LieWrapperFixture, TestLieInterfaceSubInplace)
{
    EXPECT_CALL(mocked(lieobj), sub_inplace(_)).Times(1);
    auto v = lieobj.sub_inplace(otherlie);
}
TEST_F(LieWrapperFixture, TestLieInterfaceMulInplace)
{
    EXPECT_CALL(mocked(lieobj), mul_inplace(_)).Times(1);
    auto v = lieobj.mul_inplace(otherlie);
}

TEST_F(LieWrapperFixture, TestLieInterfaceSMulInplace)
{
    EXPECT_CALL(mocked(lieobj), smul_inplace(_)).Times(1);
    auto v = lieobj.smul_inplace(scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceSDivInplace)
{
    EXPECT_CALL(mocked(lieobj), sdiv_inplace(_)).Times(1);
    auto v = lieobj.sdiv_inplace(scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceAddScalMul)
{
    EXPECT_CALL(mocked(lieobj), add_scal_mul(_, _)).Times(1);
    auto v = lieobj.add_scal_mul(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceSubScalMul)
{
    EXPECT_CALL(mocked(lieobj), add_scal_mul(_, _)).Times(1);
    auto v = lieobj.add_scal_mul(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceAddScalDiv)
{
    EXPECT_CALL(mocked(lieobj), add_scal_div(_, _)).Times(1);
    auto v = lieobj.add_scal_div(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceSubScalDiv)
{
    EXPECT_CALL(mocked(lieobj), add_scal_div(_, _)).Times(1);
    auto v = lieobj.add_scal_div(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceAddMul)
{
    EXPECT_CALL(mocked(lieobj), add_mul(_,_)).Times(1);
    auto other = setup_mock_lie();
    auto v = lieobj.add_mul(otherlie, other);
}
TEST_F(LieWrapperFixture, TestLieInterfaceSubMul)
{
    EXPECT_CALL(mocked(lieobj), sub_mul(_,_)).Times(1);
    auto other = setup_mock_lie();
    auto v = lieobj.sub_mul(otherlie, other);
}
TEST_F(LieWrapperFixture, TestLieInterfaceMulSMul)
{
    EXPECT_CALL(mocked(lieobj), mul_smul(_, _)).Times(1);
    auto v = lieobj.mul_smul(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfaceMulSDiv)
{
    EXPECT_CALL(mocked(lieobj), mul_sdiv(_, _)).Times(1);
    auto v = lieobj.mul_sdiv(otherlie,scalars::Scalar(1.0));
}
TEST_F(LieWrapperFixture, TestLieInterfacePrint)
{
    EXPECT_CALL(mocked(lieobj), print(_)).Times(1);
    std::stringstream ss;
    lieobj.print(ss);
}

TEST_F(LieWrapperFixture, TestLieInterfaceEqualsTrue)
{
    EXPECT_CALL(mocked(lieobj), equals(_))
            .Times(1)
            .WillOnce(Return(true));
    ASSERT_EQ(lieobj, otherlie);
}
TEST_F(LieWrapperFixture, TestLieInterfaceEqualsFalse)
{
    EXPECT_CALL(mocked(lieobj), equals(_))
            .Times(1);
    ASSERT_NE(lieobj, otherlie);
}
