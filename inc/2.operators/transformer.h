#pragma once

#include "operator_base.h"

// variant projection op: tensor<N, L, I> -> tensor<N, M, L, O>
class Transformer : public UnOp
{
private:
    uint _input_dim;
    uint _output_dim;
    uint _multi_head;
    bool _has_bias;

    // parameters
    Tensor _wq, _wk, _wv, _bq, _bk, _bv;

public:
    Transformer(uint input_dim, uint output_dim, uint multi_head = 1, bool has_bias = true, TensorInit_Types init_type = TensorInit_Types::Gaussian)
        : _input_dim(input_dim), _output_dim(output_dim), _multi_head(multi_head), _has_bias(has_bias)
    {
        _wq = create_param("wq", {multi_head, output_dim, input_dim}, init_type);
        _wk = create_param("wk", {multi_head, output_dim, input_dim}, init_type);
        _wv = create_param("wv", {multi_head, output_dim, input_dim}, init_type);

        if (_has_bias)
        {
            _bq = create_param("bq", {output_dim}, init_type);
            _bk = create_param("bk", {output_dim}, init_type);
            _bv = create_param("bv", {output_dim}, init_type);
        }
    }

    /*
    input: x[batch_size, node_len, input_dim], is the whole node sequence with fixed length after padding
    q, k, v[batch_size, multi_head, node_len, output_dim]
    output: y[batch_size, multi_head, node_len, output_dim]
    */
    virtual void forward(const Tensor &x, Tensor &y) const override
    {
        assert(x.shape() == 3);
        assert(x.dim()[2] == _input_dim);

        // x:[batch_size, node_len, input_dim].dot(wq/wk/wv:[multi_head, output_dim, input_dim])
        // -> q/k/v: [batch_size, node_len, multi_head, output_dim]
        Tensor q = x.dot(_wq);
        Tensor k = x.dot(_wk);
        Tensor v = x.dot(_wv);
        if (_has_bias)
        {
            q.add_(_bq);
            k.add_(_bk);
            v.add_(_bv);
        }

        q = q.swap(1, 2);           // [batch_size, multi_head, node_len, output_dim]
        k = k.swap(1, 2);           // [batch_size, multi_head, node_len, output_dim]
        v = v.move_forward(2, 2, 1); // [batch_size, multi_head, output_dim, node_len]

        // q: [batch_size, multi_head, node_len, output_dim].dot(k: [batch_size, multi_head, node_len, output_dim])
        // -> weights: [batch_size, multi_head, node_len, node_len]
        Tensor weights = q.dot(k, 2, 1); // [batch_size, multi_head, node_len, node_len]
        weights.softmax_(1);     // on last dim, [batch_size, multi_head, node_len, node_len]
        // each node i: y[i] = sum(j, weights[i][j] * v[j])
        // weights: [batch_size, multi_head, node_len, node_len].dot(v: [batch_size, multi_head, output_dim, node_len])
        //    -> y: [batch_size, multi_head, node_len, output_dim]
        y = weights.dot(v, 2, 1);
    }
};
