#pragma once
#include "unit_test.h"
#include "../inc/2.operators/fc.h"

class TestFc: public TestClass
{
public:
    REGISTER_TEST_CASES(test_forward)

    static void test_forward()
    {
        Fc op(3, 2, true, TensorInit_Types::One, TensorInit_Types::One);
        Tensor x({2, 3}, TensorInit_Types::Ordinal), y;

        op.forward(x, y);
        assert(y.dim().equals_to({2, 2}));
        assert(y.data().vector().equals_to({4, 4, 13, 13}));
    }
};