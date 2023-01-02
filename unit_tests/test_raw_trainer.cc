#pragma once
#include "unit_test.h"
#include "../inc/4.train/raw_trainer.h"

class TestRawTrainer: public TestClass
{
public:
    REGISTER_TEST_CASES(test)

    static void test()
    {
        RawTrainer trainer;
        Network network;
        Tensor x({2, 2, 3, 3}, TensorInit_Types::Ordinal);
        Tensor z({2}, TensorInit_Types::Ordinal);

        ENABLE_LOGGING(false);
        Vector<double> losses;
        trainer.train(x, z, network, {Optimizer_Types::Sgd, 0.000001, 0, 0.2}, {2, 100}, losses);
        LOG_INFO("Final loss:" << losses[losses.size() - 1]);
        // TODO: verify each step output is correct
    }
};