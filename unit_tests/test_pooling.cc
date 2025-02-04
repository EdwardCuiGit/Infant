#pragma once
#include "unit_test.h"
#include "../inc/2.operators/pooling.h"

class TestPooling: public TestClass
{
public:
    REGISTER_TEST_CASES(test_forward)

    static void test_forward()
    {
        Pooling2d op(Pooling_Types::Avg, 3, 3, 1, 1, 0, 0);
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

        assert(y.data().vector().equals_to({5, 6, 9, 10, 21, 22, 25, 26}));

        Pooling2d op1(Pooling_Types::Max, 3, 3, 1, 1, 0, 0);
        Tensor x1({1, 2, 4, 4}, TensorInit_Types::Ordinal), y1;
        y1 = op1.forward(x1);
        assert(y1.data().vector().equals_to({10, 11, 14, 15, 26, 27, 30, 31}));

        Pooling2d op2(Pooling_Types::Min, 3, 3, 1, 1, 0, 0);
        Tensor x2({1, 2, 4, 4}, TensorInit_Types::Ordinal), y2;
        y2 = op2.forward(x2);
        assert(y2.data().vector().equals_to({0, 1, 4, 5, 16, 17, 20, 21}));
    }
};