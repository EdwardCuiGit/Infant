#pragma once

#include "operator_base.h"

// projection op: vector[N, I] -> vector[N, O]

//NOTE: each Op needs to be registered in raw_trainer::init_env()
class Fc: public UnOp
{
    friend class TestOperatorBase;
public:
    struct Config : ConfigBase
    {
        DEFINE_FIELD(uint, input_dim, 1);
        DEFINE_FIELD(uint, output_dim, 1);
        DEFINE_FIELD(bool, has_bias, false);
        DEFINE_FIELD(uint, w_type, (uint)TensorInit_Types::Gaussian);
        DEFINE_FIELD(uint, b_type, (uint)TensorInit_Types::Zero);

        Config(uint _input_dim = 1, uint _output_dim = 1, bool _has_bias = 1, TensorInit_Types _w_type = TensorInit_Types::Gaussian,
        TensorInit_Types _b_type = TensorInit_Types::Zero) : ConfigBase("Fc")
        {
            input_dim() = _input_dim;
            output_dim() = _output_dim;
            has_bias() = _has_bias;
            w_type() = (uint)_w_type;
            b_type() = (uint)_b_type;
        }
    };

private:
    Config _c;

    Tensor _w, _b;

public:
    // note: _c must pass to UnOp, and c must pass to _c
    Fc(const Config & c) : UnOp(&_c, "Fc"), _c(c)
    {
        assert(_c.input_dim() > 0 && _c.output_dim() > 0);

        _w = add_param("w", {_c.output_dim(), _c.input_dim()}, (TensorInit_Types)_c.w_type());
        if (_c.has_bias())
        {
            _b = add_param("b", {_c.output_dim()}, (TensorInit_Types)_c.b_type());
        }
    }

    /*
    input:  x[batch_size, ..., input_dim]
    output: y[batch_size, ..., output_dim], will init inside the func
    y[k, i] = sum(j, x[k, j] * w[i, j])
    */
    virtual Tensor forward(const Tensor &x) const override
    {
        assert(x.shape() >= 2);
        assert(x.dim().back() == _c.input_dim());

        //[batch_size, input_dim] . [output_dim, input_dim] => [batch_size, output_dim]
        Tensor y = x.dot(_w); // dot-product between a batch of vectors and a list of weights
        if (_c.has_bias())
        {
            y.add_(_b); // add to y
        }

        //y = y.squeeze(); // note: this is not good if it's expected 1 size dim
        return y;
    }
};