#pragma once

#include "../2.operators/operator_base.h"
#include "../2.operators/fc.h"
#include "../2.operators/conv.h"
#include "../2.operators/norm.h"
#include "../2.operators/pooling.h"
#include "../2.operators/rnn.h"
#include "../2.operators/transformer.h"
#include "../1.functors/functor_graph.h"

// TODO: for visualization & debugging purpose, we'd better build the operator graph to know each operator's input & output & parameters
class Network// : public FunctorGraph
{
public:
    // This forward function is used to build network graph w/ auto_grad
    // Note: this is one example code to create a new network
    // TODO: build graph without create Tensor data & first iteration forward
    virtual TensorList build_network_core(const TensorList &x)
    {
        // below are sample code
        assert(x.size() == 2);
        assert(x[0].dim().equals_to({2, 2, 3, 3})); // N, C, H, W
        Tensor y1 = x[0].conv2d(2, 3, 2, 2, 1, 1, 0, 0, 1, false, TensorInit_Types::Gaussian);
        Tensor y2 = y1.pool2d(Pooling_Types::Avg, 2, 2);
        Tensor y3 = y2.fc(3, 1, true, TensorInit_Types::Ordinal, TensorInit_Types::Gaussian);
        y3.set_print();

        assert(x[1].dim().equals_to({2}));
        Tensor y4 = x[1].mse(y3);
        assert(y4.dim().equals_to({1}));
        return {y4};
    }

    TensorList build_network(const TensorList& x)
    {
        FunctorGraph::singleton().set_inputs(x);
        auto y = this->build_network_core(x);
        FunctorGraph::singleton().set_outputs(y);
        return y;
    }

    void set_train(bool yes = true)
    {
        FunctorGraph::singleton().set_train(yes);
    }

    TensorList forward(const TensorList &x) const
    {
        auto y = FunctorGraph::singleton().forward(x);
        FunctorGraph::singleton().print(std::cout);
        return y;
    }

    double calc_loss(const Tensor& y0) const
    {
        assert(y0.size() > 0);
        FunctorGraph::singleton().stop_auto_grad();
        double loss = y0.avg().data().first_item();
        FunctorGraph::singleton().start_auto_grad();
        return loss;
    }

    void backward(const TensorList &y) const
    {
        FunctorGraph::singleton().backward(y);
    }

    void zero_features() const
    {
        FunctorGraph::singleton().zero_features();
    }

    void zero_grads() const
    {
        FunctorGraph::singleton().zero_grads();
    }

    TensorList& params()
    {
        return FunctorGraph::singleton().params();
    }

    void reset()
    {
        FunctorGraph::singleton().reset();
    }
};