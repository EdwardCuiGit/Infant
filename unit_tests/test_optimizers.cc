#pragma once
#include "unit_test.h"
#include "../inc/4.train/optimizer_base.h"

class TestOptimizers: public TestClass
{
public:
    REGISTER_TEST_CASES(test_sgd)

    static void test_sgd()
    {
        Sgd sgd(0.1, 0.2, 0.0);
        TensorList params;
        Tensor p1;
        p1.reset({2, 3}, TensorInit_Types::One);
        p1.grad().reset({2, 3}, TensorInit_Types::Ordinal);
        params.push_back(p1);
        sgd.step(params);
        assert(p1.data().vector().equals_to({1, 1-0.1, 1-0.2, 1-0.3, 1-0.4, 1-0.5}));

        sgd.step(params);
        assert(p1.data().vector().equals_to({1, 1-0.1*(2+0.2), 1-0.2*(2+0.2), 1-0.3*(2+0.2), 1-0.4*(2+0.2), 1-0.5*(2+0.2)}));
    }
};