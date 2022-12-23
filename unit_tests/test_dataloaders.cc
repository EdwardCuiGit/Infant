#pragma once
#include "unit_test.h"
#include "../inc/3.network/data_loader.h"

class TestDataLoaders: public TestClass
{
public:
    REGISTER_TEST_CASES(test_inmemory_dataloader)

    static void test_inmemory_dataloader()
    {
        Tensor x({5, 2, 2}, TensorInit_Types::Ordinal), batch;
        InMemoryDataLoader dl(x);

        uint samples = dl.read_next(batch, 2);
        assert(samples == 2);
        assert(batch.data().vector().equals_to({0, 1, 2, 3, 4, 5, 6, 7}));
        assert(batch.dim().equals_to({2, 2, 2}));

        samples = dl.read_next(batch, 2);
        assert(samples == 2);
        assert(batch.data().vector().equals_to({8, 9, 10, 11, 12, 13, 14, 15}));
        assert(batch.dim().equals_to({2, 2, 2}));

        samples = dl.read_next(batch, 2);
        assert(samples == 1);
        assert(batch.data().vector().equals_to({16, 17, 18, 19}));
        assert(batch.dim().equals_to({1, 2, 2}));

        samples = dl.read_next(batch, 2);
        assert(samples == 0);
        assert(batch.data().vector().equals_to({16, 17, 18, 19}));
        assert(batch.dim().equals_to({1, 2, 2}));
    }
};