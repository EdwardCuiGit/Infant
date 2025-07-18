#pragma once
#include "functor.h"

// BinFunctor has 2 Tensor inputs, 1 tensor output
class BinFunctor : public Functor
{
protected:
    uint _first_match_dims;
    int _last_work_dims;

    BinFunctor(const std::string& type, uint first_match_dims, int last_work_dims)
        : Functor(type), _first_match_dims(first_match_dims), _last_work_dims(last_work_dims)
    {
    }

public:
    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const = 0;

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                          TensorD<float> &x1_grad, TensorD<float> &x2_grad) const = 0;

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint input_tensor_count() const override
    {
        return 2;
    }

    virtual uint output_tensor_count() const override
    {
        return 1;
    }
};

// could support in-place, since x is not used
class Add : public BinFunctor
{
protected:
    float _alpha_x1;
    float _alpha_x2;
    float _beta;

public:
    Add(float alpha_x1, float alpha_x2, float beta, uint first_match_dims, int last_work_dims)
        : BinFunctor("Add", first_match_dims, last_work_dims), _alpha_x1(alpha_x1), _alpha_x2(alpha_x2), _beta(beta)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.add(x2, y, _alpha_x1, _alpha_x2, _beta, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // note: x1, x2, y is not used, only using alpha, save memory if possible
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.add_grad(x2, y, y_grad, x1_grad, x2_grad, true, _alpha_x1, _alpha_x2, _beta, _first_match_dims, _last_work_dims);
    }
};

class Mul : public BinFunctor
{
protected:
    float _alpha;
    float _beta;

public:
    Mul(float alpha, float beta, uint first_match_dims, int last_work_dims)
        : BinFunctor("Mul", first_match_dims, last_work_dims), _alpha(alpha), _beta(beta)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.mul(x2, y, _alpha, _beta, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // note: x1, x2, y is not used, only using alpha, save memory if possible
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.mul_grad(x2, y, y_grad, x1_grad, x2_grad, _alpha, _beta, true, _first_match_dims, _last_work_dims);
    }
};

class Dot : public BinFunctor
{
public:
    Dot(uint first_match_dims, int last_work_dims) : BinFunctor("Dot", first_match_dims, last_work_dims)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.dot(x2, y, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // note: y is not used
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.dot_grad(x2, y, y_grad, x1_grad, x2_grad, true, _first_match_dims, _last_work_dims);
    }
};

// below are usually used to calculate loss
class Mse : public BinFunctor
{
public:
    Mse(uint first_match_dims, int last_work_dims) : BinFunctor("Mse", first_match_dims, last_work_dims)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.mse(x2, y, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // note: y is not used
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.mse_grad(x2, y, y_grad, x1_grad, x2_grad, true, _first_match_dims, _last_work_dims);
    }
};

class Ce : public BinFunctor
{
public:
    Ce(uint first_match_dims, int last_work_dims) : BinFunctor("Ce", first_match_dims, last_work_dims)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.ce(x2, y, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.ce_grad(x2, y, y_grad, x1_grad, x2_grad, true, _first_match_dims, _last_work_dims);
    }
};

class Euclidean : public BinFunctor
{
public:
    Euclidean(uint first_match_dims, int last_work_dims) : BinFunctor("Euclidean", first_match_dims, last_work_dims)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.euclidean(x2, y, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.euclidean_grad(x2, y, y_grad, x1_grad, x2_grad, true, _first_match_dims, _last_work_dims);
    }
};

class Append : public BinFunctor
{
    uint _dim_to_inc;
public:
    Append(uint dim_to_inc = 0) : BinFunctor("Append", 0, 0), _dim_to_inc(dim_to_inc)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.append(x2, y, _dim_to_inc);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        x1.append_grad(x2, y, y_grad, x1_grad, x2_grad, true, _dim_to_inc);
    }

};

class EncodeByDict: public BinFunctor
{
public:
    EncodeByDict() : BinFunctor("EncodeByDict", 0, 0)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.encode_by_dict(x2, y);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // x1_grad is input tensor, no need to calc grad
        x1.encode_by_dict_grad(x2, y, y_grad, x2_grad);
    }
};

class SearchByDict: public BinFunctor
{
    int _padding_id;
public:
    SearchByDict(int padding_id = -1) : BinFunctor("SearchByDict", 0, 0), _padding_id(padding_id)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.search_by_dict(x2, y, _padding_id);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // x1_grad is target tensor, no need to calc grad
        x1.search_by_dict_grad(x2, y, y_grad, x2_grad, _padding_id);
    }
};  

class Merge: public BinFunctor
{
    int _dim;
public:
    Merge(int dim = 0) : BinFunctor("Merge", 0, 0), _dim(dim)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.merge(x2, y, _dim);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        x1.merge_grad(x2, y, y_grad, x1_grad, x2_grad, _dim);
    }
};


/*
class Encode: public BinFunctor
{
public:
    Encode(uint first_match_dims, int last_work_dims) : BinFunctor("Encode", first_match_dims, last_work_dims)
    {
    }

    virtual void forward(const TensorD<float> &x1, const TensorD<float> &x2, TensorD<float> &y) const override
    {
        x1.encode(x2, y, _first_match_dims, _last_work_dims);
    }

    virtual void backward(const TensorD<float> &x1, const TensorD<float> &x2, const TensorD<float> &y, const TensorD<float> &y_grad,
                           TensorD<float> &x1_grad, TensorD<float> &x2_grad) const override
    {
        // TODO: dynamic confirm whether x2_grad needs to calc
        x1.encode_grad(x2, y, y_grad, x1_grad, x2_grad, true, _first_match_dims, _last_work_dims);
    }
};*/