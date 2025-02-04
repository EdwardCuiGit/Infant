#pragma once
#include "unit_test.h"
#include "../inc/4.train/trainers.h"

class TestTrainers: public TestClass
{
public:
    REGISTER_TEST_CASES(test_sample_cnn, test_transformer)

    static void test_sample_cnn()
    {
        SampleCnnTrainer trainer;
        Tensor x({2, 2, 3, 3}, TensorInit_Types::Ordinal);
        Tensor z({2}, TensorInit_Types::Ordinal);

        ENABLE_LOGGING(true);
        trainer.train(x, z, TrainerConfig());
        // TODO: verify each step output is correct
    }

    static void test_transformer()
    {
        TransformerTrainer trainer;
        trainer.train();
        // TODO: verify each step output is correct
    }
};