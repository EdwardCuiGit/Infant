// This class no need to exist, since we could make it more generic by run backward() from any tensor
#ifdef __UNDEFINED__
#pragma once

#include "operators/operator_base.h"

// TODO: each sample may have a weight
class LossBase : public Operator
{
protected:
    double _w; // not init yet TODO
    mutable double _loss;
    TensorList _empty;

public:
    OVERRIDE bool has_loss() const
    {
        return true;
    }

    double get_loss() const
    {
        return _loss;
    }

    // input: y is loss input, retrieved from NetworkNode->input_nodes[0]
    // input: t is ground truth, retrieved from NetworkNode->input_nodes[1]
    // output: y_loss is loss value per sample, stored in NetworkNode->outputs[0]
    // output: returns overall loss for this batch, print to log output
    virtual void forward(const TensorD<double> &y, const TensorD<double> &t, TensorD<double> &y_loss) const = 0;

    // this is loss function's gradient calc function
    // input: y is loss input, retrieved from NetworkNode->input_nodes[0]
    // input: t is ground truth, retrieved from NetworkNode->input_nodes[1]
    // input: y_loss is loss value per sample, retrieved from NetworkNode->outputs[0]
    // output: y_grad: stored in NetworkNode->input_grads[0]
    virtual void backward(const TensorD<double> &y, const TensorD<double> &t, const TensorD<double> &y_loss, TensorD<double> &y_grad) = 0;

    OVERRIDE void forward(const TensorList &x, TensorList &y) const
    {
        assert(x.size() == 2);
        y.reserve(1);
        this->forward(x[0].data(), x[1].data(), y[0].data());
    }

    // y_grad is not assigned & not used
    OVERRIDE void backward(TensorList &x, const TensorList &y)
    {
        assert(x.size() == 2);
        assert(y.size() == 1);
        this->backward(x[0].data(), x[1].data(), y[0].data(), x[0].grad());
    }
};

class MseLoss : public LossBase
{
    // input: y: [..., batch_size]
    // input: t: [..., batch_size]
    // output: y_loss: [..., batch_size]
    // output: l: float
    // mse loss: l = avg((y - t)^2)
    OVERRIDE void forward(const TensorD<double> &y, const TensorD<double> &t, TensorD<double> &y_loss) const
    {
        assert(y.shape() >= 1);
        assert(y.dim().equals_to(t.dim()));

        y.mse(t, y_loss, y.shape() - 1, 1);
        _loss = y_loss.avg();
    }

    // mse loss's grad: x_grad = dy/dx = 2(x - t)/N = 2/N x - 2N t
    // x: [..., batch_size]
    // y_grad: not used, empty tensor since not assigned
    // x_grad: [..., batch_size]
    OVERRIDE void backward(const TensorD<double> &y, const TensorD<double> &t, const TensorD<double> &y_loss, TensorD<double> &y_grad)
    {
        uint n = t.size();
        if (n == 0)
            return;
        y.add(t, y_grad, 0, -1, false, 2.0 / n, -2.0 / n, 0.0);
    }
};

// TODO: supports one-hot ce
// TODO: supports binary one-hot ce
class CrossEntropyLoss : public LossBase
{
    // input:  y[.., batch_size, vocab_size]
    // input:  t[.., batch_size, vocab_size], each value is uint which is target ground truth id
    // output: y_loss[.., batch_size], each value is loss per sample
    // output: l: float, one aggregated value for whole batch's loss
    OVERRIDE void forward(const TensorD<double> &y, const TensorD<double> &t, TensorD<double> &y_loss) const
    {
        assert(y.shape() >= 2);
        assert(y.dim().equals_to(t.dim()));
        assert(y.dim()[y.shape() - 2] == t.dim()[y.shape() - 1]);

        y.ce(t, y_loss, y.shape() - 1, 1);
        _loss = y_loss.avg();
    }

    OVERRIDE void backward(const TensorD<double> &y, const TensorD<double> &t, const TensorD<double> &y_loss, TensorD<double> &y_grad)
    {
        assert(false); // TODO
    }
};
#endif