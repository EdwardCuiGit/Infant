#pragma once

#include "operator_base.h"

// projection op: vector[N, I] -> vector[N, O]
class Fc: public UnOp
{
private:
    uint _input_dim;
    uint _output_dim;
    bool _has_bias;

    Tensor _w, _b;

public:
    Fc(uint input_dim, uint output_dim, bool has_bias = true, TensorInit_Types w_type = TensorInit_Types::Gaussian,
    TensorInit_Types b_type = TensorInit_Types::Zero)
        : _input_dim(input_dim), _output_dim(output_dim), _has_bias(has_bias)
    {
        assert(input_dim > 0 && output_dim > 0);

        _w = create_param("w", {output_dim, input_dim}, w_type);
        if (_has_bias)
        {
            _b = create_param("b", {output_dim}, b_type);
        }
    }

    /*
    input:  x[batch_size, input_dim]
    output: y[batch_size, output_dim], will init inside the func
    y[k, i] = sum(j, x[k, j] * w[i, j])
    */
    virtual void forward(const Tensor &x, Tensor &y) const override
    {
        assert(x.shape() == 2);
        assert(x.dim().back() == _input_dim);

        //[batch_size, input_dim] . [output_dim, input_dim] => [batch_size, output_dim]
        y = x.dot(_w); // dot-product between a batch of vectors and a list of weights
        if (_has_bias)
        {
            y.add_(_b); // add to y
        }

        y = y.squeeze();
    }
};

// projection op: vector[N, L, I] -> vector[N, L, O]
// TODO: forward func is the same as Linear, backward may be different

class NodeLinear : public Linear
{
};