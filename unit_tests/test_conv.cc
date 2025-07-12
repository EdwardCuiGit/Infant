#pragma once
#include "unit_test.h"
#include "inc/2.operators/conv.h"

class TestConv: public TestClass
{
public:
    REGISTER_TEST_CASES(test_forward)

    static void test_forward()
    {
        Conv2d op(2, 2, 3, 3, 1, 1, 0, 0, 1, true, TensorInit_Types::One, TensorInit_Types::One);
        Tensor x({1, 2, 4, 4}, TensorInit_Types::Ordinal), y;
        /*
        channel 0:
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
        channel 1:
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
        */

        y = op.forward(x);

        assert(y.dim().equals_to({1, 2, 2, 2}));
        assert(y.data().vector().equals_to({234+1, 234+18+1, 306+1, 306+18+1, 234+1, 234+18+1, 306+1, 306+18+1}));
    }
};