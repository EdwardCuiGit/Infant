#pragma once
#include "unit_test.h"
#include "inc/trees/gbdt.h"

class TestGbdt: public TestClass
{
public:
    REGISTER_TEST_CASES(test_train, test_inference)

    static void test_train()
    {
        Gbdt gbdt;
        Gbdt::GbdtEnsemble model;
        gbdt.train({}, {}, {}, model);
    }

    static void test_inference()
    {
        Gbdt gbdt;
        Gbdt::GbdtEnsemble model;
        TensorD<float> y;
        gbdt.inference({}, y, model);
    }
};