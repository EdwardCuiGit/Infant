#pragma once

#include "inc/2.operators/fc.h"
#include "inc/2.operators/conv.h"
#include "inc/2.operators/pooling.h"

class SampleCnn: public Operator
{
private:
    Ptr<Conv2d> _conv;
    Ptr<Pooling2d> _pool;
    Ptr<Fc> _fc;

public:
    SampleCnn() : Operator(nullptr, "SampleCnn")
    {
        // build operator graph by initiating all the operators
        _conv = std::make_shared<Conv2d>(2, 3, 2, 2, 1, 1, 0, 0, 1, false, TensorInit_Types::Gaussian);
        _pool = std::make_shared<Pooling2d>(Pooling_Types::Avg, 2, 2);
        _fc = std::make_shared<Fc>(Fc::Config{3, 1, true, TensorInit_Types::Ordinal, TensorInit_Types::Gaussian});
    }

    virtual TensorList forward(const TensorList &x) const override
    {
        assert(x.size() == 2);
        assert(x[0].dim().equals_to({2, 2, 3, 3})); // N, C, H, W
        Tensor y1 = _conv->forward(x[0]);
        Tensor y2 = _pool->forward(y1);
        Tensor y3 = _fc->forward(y2).merge_dim(0, 2);
        y3.set_print();

        assert(x[1].dim().equals_to({2}));
        Tensor y4 = x[1].mse(y3);
        assert(y4.dim().equals_to({1}));
        return {y4};
    }
};