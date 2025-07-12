#pragma once
#include "unit_test.h"
#include "inc/4.train/data_loader.h"

class TestDataLoaders: public TestClass
{
public:
    REGISTER_TEST_CASES(test_inmemory_dataloader, test_text_file_loader)

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

    static void test_text_file_loader()
    {
        return;
        /* test.txt fil content
        123
        12
        2
        */ 
        TextFileLoader tfl("test.txt", 3);
        Tensor data;
        uint samples = tfl.read_next(data, 4);
        assert(samples == 3);
        assert(data.data().vector().equals_to({11, 12, 2, 11, 12, 2, 12, 2, 0}));
        assert(data.dim().equals_to({3, 3}));
        tfl.reset();
    }
};