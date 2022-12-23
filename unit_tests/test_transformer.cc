#pragma once
#include "unit_test.h"
#include "../inc/2.operators/transformer.h"

class TestTransformer: public TestClass
{
public:
    REGISTER_TEST_CASES(test_forward)

    static void test_forward()
    {
        // tensor<N, L, I> -> tensor<N, M, L, O>: 2, 2, 3, 1
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal), y;
        /*
        inputs: seq-0: 0, 1,  2, 3,  4, 5, seq-1: 6, 7,  8, 9,  10,11
        q/k/v: 1, 5, 9; 13, 17, 21
        q*k: 1, 5, 9; 5, 25, 45; 9, 45, 81; 13*13, 13*17, 13*21; 17*13, 17*17, 17*21, 21 * 13, 21*17, 21*21;
        ...
        */
        Transformer op(2, 1, 2, false, TensorInit_Types::One);

        // softmax_ results incorrect, dot dim() incorrect
        op.forward(x, y);
        assert(y.dim().equals_to({2, 2, 3, 1}));
        assert(y.data().vector().equals_to({8.925444, 8.999999, 9, 8.925444, 8.999999, 9, 21, 21, 21, 21, 21, 21}));
        LOG_WARN("TestTransformer:checked 1 case, but not fully verify results yet");
    }
};