#pragma once
#include "unit_test.h"
#include "../inc/2.operators/rnn.h"

class TestRnn: public TestClass
{
public:
    REGISTER_TEST_CASES(test_RawRnn_forward, test_Lstm_forward, test_Gru_forward)

    static void test_RawRnn_forward()
    {
        // o = act2(act1(x * w + h * u + b1) * v + b2)
        RawRnn rnn(2, 3, 1, false, TensorInit_Types::One, Activation_Types::Linear, Activation_Types::Linear);
        Tensor x({1, 3, 2}, TensorInit_Types::Ordinal);
        /*
        seq-0: 0,1,  2,3,  4,5
        */
        TensorList y;

        rnn.forward({x}, y);
        assert(y.size() == 2);
        assert(y[1].dim().equals_to({1, 3, 1}));
        assert(y[1].data().vector().equals_to({3, 24, 99}));
    }

    static void test_Lstm_forward()
    {
        // f = sigmoid(x * _wf + h * _uf + bf) => 1,1,  5+2,7,     9+14*56, 9+14*56
        // i = sigmoid(x * _wi + h * _ui + bi) => 1,1
        // o = sigmoid(x * _wo + h * _uo + bo) => 1,1
        // c = sigmoid(x * _wc + h * _uc + bc) => 1,1
        // out = f mul out + i mul c           => 1,1, 7+49,56,    (9+14*56)*(56 + 9+14*56), ..
        // h = o mul sigmoid(out)              => 1,1, 7*56, 7*56, (9+14*56)^2*(56+9+14*56), ..
        Lstm rnn(2, 2, false, TensorInit_Types::One, Activation_Types::Linear, Activation_Types::Linear);
        Tensor x({1, 3, 2}, TensorInit_Types::Ordinal);
        /*
        seq-0: 0,1,  2,3,  4,5
        */
        TensorList y;

        rnn.forward({x}, y);
        assert(y.size() == 2);
        assert(y[1].dim().equals_to({1, 3, 2}));
        assert(y[1].data().vector().equals_to({1, 1, 7*56, 7*56, (9+14*56)*(9+14*56)*(56+9+14*56), (9+14*56)*(9+14*56)*(56+9+14*56)}));
    }

    static void test_Gru_forward()
    {
        // z = sigmoid(x * _wz + h * _uz + bz)         1,1,  7,7,            9+86, 9+86,
        // r = sigmoid(x * _wr + h * _ur + br)         1,1,  7,7
        // h_origin = sigmoid(x * _wh + h * _uh + bh)  1,1,  7,7
        // h = h mul (1 - z) + h_origin mul z          1,1,  49-6=43, 49-6,  -94*43 + (9+86)*(9+86)

        Gru rnn(2, 2, false, TensorInit_Types::One, Activation_Types::Linear, Activation_Types::Linear);
        Tensor x({1, 3, 2}, TensorInit_Types::Ordinal);
        /*
        seq-0: 0,1,  2,3,  4,5
        */
        TensorList y;

        rnn.forward({x}, y);
        assert(y.size() == 2);
        assert(y[1].dim().equals_to({1, 3, 2}));
        assert(y[1].data().vector().equals_to({1, 1, 43, 43, 95*95-94*43, 95*95-94*43}));
    }
};