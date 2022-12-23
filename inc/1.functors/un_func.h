#pragma once

#include "unmo_func.h"

// Unary Math Operator, one input tensor, one output tensor
// supported functors: Linear, Pow, Softamx, Activation, Sum, Avg, Var, Max, Min, Swap, MoveForward, Im2Col
// MergeDim, Squeeze
class UnFunctor : public UnMoFunctor
{
public:
    UnFunctor(const std::string& type, int last_work_dims = -1)
        : UnMoFunctor(type, last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const = 0;

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const = 0;

    virtual void forward(const TensorD<double> &x, TensorList &y) const override;

    virtual void backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const override;

    virtual uint output_tensor_count() const override
    {
        return 1;
    }
};

//note: most below have similar code structure, how shall we make them simple 
class Linear : public UnFunctor
{
private:
    double _alpha;
    double _beta;

public:
    Linear(double alpha = 1.0, double beta = 0.0, int last_work_dims = -1)
        : UnFunctor("Linear", last_work_dims), _alpha(alpha), _beta(beta)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.linear(y, _alpha, _beta, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override 
    {
        x.linear_grad(y, y_grad, x_grad, _alpha, _beta, _last_work_dims);
    }
};

class Pow : public UnFunctor
{
private:
    double _n, _bias;

public:
    Pow(double n = 1.0, double bias = 0, int last_work_dims = -1)
        : UnFunctor("Pow", last_work_dims), _n(n), _bias(bias)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.pow(y, _n, _bias, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.pow_grad(y, y_grad, x_grad, _n, _bias, _last_work_dims);
    }
};

class Softmax : public UnFunctor
{
private:
public:
    Softmax(int last_work_dims = -1)
        : UnFunctor("Softmax", last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.softmax(y, _last_work_dims);
    }

    // note: x, y may not be used: we could collect them and free them in advance
    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.softmax_grad(y, y_grad, x_grad, _last_work_dims);
    }
};

class Activation : public UnFunctor
{
private:
    Activation_Types _act_type;

public:
    Activation(Activation_Types act_type, int last_work_dims = -1)
        : UnFunctor("Activation", last_work_dims), _act_type(act_type)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.activation(_act_type, y, _last_work_dims);
    }

    // TODO: not used: x, y: we could collect them and free them in advance
    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.activation_grad(_act_type, y, y_grad, x_grad, _last_work_dims);
    }
};


class Sum : public UnFunctor
{
public:
    Sum(int last_work_dims = -1)
        : UnFunctor("Sum", last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.sum(y, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.sum_grad(y, y_grad, x_grad, _last_work_dims);
    }
};

class Avg : public UnFunctor
{
public:
    Avg(int last_work_dims = -1)
        : UnFunctor("Avg", last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.avg(y, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.avg_grad(y, y_grad, x_grad, _last_work_dims);
    }
};

class Var : public UnFunctor
{
private:
    bool _biased;

public:
    Var(bool biased = false, int last_work_dims = -1)
        : UnFunctor("Var", last_work_dims), _biased(biased)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.var(y, _biased, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.var_grad(y, y_grad, x_grad, _biased, _last_work_dims);
    }
};


class Max : public UnFunctor
{
public:
    Max(int last_work_dims = -1)
        : UnFunctor("Max", last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.max(y, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.max_grad(y, y_grad, x_grad, _last_work_dims);
    }
};

class Min : public UnFunctor
{
public:
    Min(int last_work_dims = -1)
        : UnFunctor("Min", last_work_dims)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.min(y, _last_work_dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.min_grad(y, y_grad, x_grad, _last_work_dims);
    }
};

class Swap : public UnFunctor
{
private:
    uint _first_dim;
    uint _second_dim;

public:
    Swap(uint first_dim, uint second_dim) : UnFunctor("Swap"), _first_dim(first_dim), _second_dim(second_dim)
    {
    }

    // Note: not support common Un parameters: _last_work_dims
    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.swap(y, _first_dim, _second_dim);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        y_grad.swap(x_grad, _first_dim, _second_dim);
    }
};

class MoveForward : public UnFunctor
{
private:
    uint _move_from, _move_len, _move_to;

public:
    // Note: not support common Un parameters: _last_work_dims
    MoveForward(uint move_from, uint move_len, uint move_to) : UnFunctor("MoveForward"), _move_from(move_from), _move_len(move_len), _move_to(move_to)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.move_forward(y, _move_from, _move_len, _move_to);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        y_grad.move_forward(x_grad, _move_to + _move_len, _move_from - _move_to, _move_to);
    }
};

class Im2Col : public UnFunctor
{
private:
    uint _groups, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y;

public:
    Im2Col(uint groups, uint kernel_x, uint kernel_y, uint stride_x, uint stride_y, uint padding_x, uint padding_y)
        : UnFunctor("Im2Col"), _groups(groups), _kernel_x(kernel_x), _kernel_y(kernel_y), _stride_x(stride_x), _stride_y(stride_y), _padding_x(padding_x), _padding_y(padding_y)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.im2col(y, _groups, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.im2col_grad(y, y_grad, x_grad, _groups, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y);
    }
};

class MergeDim: public UnFunctor
{
private:
    uint _from, _len;
public:
    MergeDim(uint from, uint len) : UnFunctor("MergeDim"), _from(from), _len(len){}

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.merge_dim(y, _from, _len);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.merge_dim_grad(y, y_grad, x_grad, _from, _len);
    }
};

class Inflate: public UnFunctor
{
private:
    const Vector<uint> _dims;
public:
    Inflate(const Vector<uint>& dims) : UnFunctor("Inflate"), _dims(dims)
    {}

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.inflate(y, _dims);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.inflate_grad(y, y_grad, x_grad, _dims);
    }
};

class Squeeze : public UnFunctor
{
public:
    Squeeze() : UnFunctor("Squeeze")
    {}

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.squeeze(y);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.squeeze_grad(y, y_grad, x_grad);
    }
};

class Subset: public UnFunctor
{
private:
    Vector<uint> _dim;
    uint _offset;
public:
    Subset(const Vector<uint>& dim, uint offset) : UnFunctor("Subset"), _dim(dim), _offset(offset)
    {}

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.subset(y, _dim, _offset);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        x.subset_grad(y, y_grad, x_grad, _dim, _offset);
    }
};

#ifdef DISABLED
// not implemented
class Dropout : public UnFunctor
{
private:
    double _p;

public:
    Dropout(double p, uint move_to) : UnFunctor(), _p(p)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        // use p probability to make one elem to be 0
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        // for the y that is not zero, just pass y_grad
    }
};

// below are map functions
class Map : public UnFunctor
{
private:
    const std::function<double(double)> &_map_func;
    const std::function<double(double, double)> &_map_grad_func;

public:
    Map(const std::function<double(double)> &map_func, const std::function<double(double, double)> &map_grad_func,
        int last_work_dims = -1, bool add_to = false, int t1_id = -1)
        : UnFunctor(last_work_dims, add_to, t1_id), _map_func(map_func), _map_grad_func(map_grad_func)
    {
    }

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.map(_map_func, y, _last_work_dims, _add_to, _t1_id);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        assert(x.size() == y_grad.size());

        x.map(_map_grad_func, y_grad, x_grad, _last_work_dims, _add_to, _t1_id);
    }
};

class MapBase : public UnFunctor
{
public:
    MapBase(int last_work_dims = -1, bool add_to = false, int t1_id = -1)
        : UnFunctor(last_work_dims, add_to, t1_id)
    {
    }

    virtual double map(double x) const = 0;

    virtual double map_grad(double x, double y_grad) const = 0;

    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const override
    {
        x.map([this](double x) -> double
              { return this->map(x); },
              y, _last_work_dims, _add_to, _t1_id);
    }

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) const override
    {
        assert(x.size() == y_grad.size());

        x.map([this](double x, double y_grad) -> double
              { return this->map_grad(x, y_grad); },
              y_grad, x_grad, _last_work_dims, _add_to, _t1_id);
    }
};

#endif