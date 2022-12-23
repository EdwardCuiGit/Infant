#pragma once
#include "functor.h"

// one input tensor, multiple output tensor
class UnMoFunctor : public Functor
{
protected:
    int _last_work_dims;

    // void (*forward_fn)(const TensorD<double> &x, TensorList &y) = nullptr;
    // void (*backward_fn)(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) = nullptr;

public:
    UnMoFunctor(const std::string& type, int last_work_dims = -1)
        : Functor(type), _last_work_dims(last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorList &y) const = 0;

    virtual void backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const = 0;

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint input_tensor_count() const override
    {
        return 1;
    }
};

// TODO: perf tuning: no need to physically allocate memories
class Divide : public UnMoFunctor
{
private:
    uint _first_match_dims;

public:
    Divide(uint first_match_dims = 1)
        : UnMoFunctor("Divide"), _first_match_dims(first_match_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorList &y) const override;

    virtual void backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const override;
};

/*class MeanVar : public UnMoFunctor
{
private:
    bool _biased;

public:
    MeanVar(bool biased = false, int last_work_dims = -1)
        : UnMoFunctor(last_work_dims), _biased(biased)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorList &y) const override;

    virtual void backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const override;

    OVERRIDE uint output_tensor_count() const
    {
        return 2;
    }
};*/

