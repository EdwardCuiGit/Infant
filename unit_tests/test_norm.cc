#pragma once
#include "unit_test.h"
#include "../inc/2.operators/norm.h"

class TestNorm: public TestClass
{
public:
    REGISTER_TEST_CASES(test_BatchNorm2d_forward, test_LayerNorm_forward)

    static void test_BatchNorm2d_forward()
    {
        BatchNorm2d norm(2, 0.5, true, true);
        Tensor x({1, 2, 1, 2}, TensorInit_Types::Ordinal), y;
        norm.forward(x, y);
        assert(y.dim().equals_to({1, 2, 1, 2}));
        assert(y.data().vector().equals_to({-1, 1, -1, 1}));
    }

    static void test_LayerNorm_forward()
    {
        LayerNorm norm({1, 2}, 0.1, true, true);
        Tensor x({1, 2, 1, 2}, TensorInit_Types::Ordinal), y;
        norm.forward(x, y);
        assert(y.dim().equals_to({1, 2, 1, 2}));
        assert(y.data().vector().equals_to({-1, 1, -1, 1}));
    }
};