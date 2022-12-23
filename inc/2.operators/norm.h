#pragma once

#include "operator_base.h"

/*
Normalization is poitwise ops: vector[N, I] -> vector[N, I]
https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8
    Why: faster training 
Softmax: in un_func.h
BatchNorm: works for [N, C, H, W], each channel has one mean/variance
    https://arxiv.org/pdf/1502.03167.pdf%27
    TODO: BatchNorm1d, BatchNorm3d
    issues: variable batch size, RNN needs a mean/variance per each time step
TODO: LayerNorm;
    https://arxiv.org/pdf/1607.06450.pdf, claims LN is better than BN in RNN
TODO: InstanceNorm;
*/

class NormBase : public UnOp
{
};

class BatchNorm2d : public NormBase
{
private:
    Tensor _alpha, _beta;
    mutable TensorD<double> _running_mean, _running_var; // assigned but not used
    mutable uint _num_batches_tracked;
    uint _channels;
    double _momentum;
    bool _affine;
    bool _track_running_stats;

public:
    BatchNorm2d(uint channels, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        : _channels(channels), _momentum(momentum), _affine(affine), _track_running_stats(track_running_stats)
    {
        assert(channels > 0);
        if (affine)
        {
            _alpha = create_param("alpha", {_channels}, TensorInit_Types::One);
            _beta = create_param("beta", {_channels}, TensorInit_Types::Zero);
        }

        if (track_running_stats)
        {
            _num_batches_tracked = 0;
            _running_mean.reset({_channels}, TensorInit_Types::Zero);
            _running_var.reset({_channels}, TensorInit_Types::One);
        }
    }

    // input: x[N, C, H, W], output: y[N, C, H, W], each channel has one mean/variance value
    // x[i] = (x[i] - mean) / sqrt(variance + e) * alpha + beta;
    virtual void forward(const Tensor &x, Tensor &y) const override
    {
        assert(x.shape() == 4);
        assert(x.dim()[1] == _channels);
        auto x_c = x.swap(0, 1);                // [C, N, H, W]
        auto mean = x_c.avg(3), var = x_c.var(false, 3); // mean/var: [C]

        // note: perf: change to one loop
        x_c.add_(mean, 1, -1, 0, 1, 0).mul_(var.pow(-0.5, EPSILON), 1, 0, 1, 0);
        if (_affine)
        {
            x_c.mul_(_alpha, 1, 0, 1, 0).add_(_beta, 1, 1, 0, 1, 0);
        }

        y = x_c.swap(0, 1);

        double exp_avg_factor = _momentum;
        if (_is_train && _track_running_stats)
        {
            _num_batches_tracked++;
            if (_momentum == 0)
            {
                exp_avg_factor = 1.0 / _num_batches_tracked;
            }

            _running_mean.add(mean.data(), _running_mean, exp_avg_factor, 1 - exp_avg_factor);
            _running_var.add(var.data(), _running_var, exp_avg_factor, 1 - exp_avg_factor);
        }
    }
};

class LayerNorm : public NormBase
{
private:
    Tensor _alpha, _beta;
    mutable TensorD<double> _running_mean, _running_var; // assigned but not used
    mutable uint _num_batches_tracked;
    Vector<uint> _last_dims;
    double _momentum;
    bool _affine;
    bool _track_running_stats;

public:
    LayerNorm(const Vector<uint>& last_dims, double momentum = 0.1, bool affine = true, bool track_running_stats = true)
        : _last_dims(last_dims), _momentum(momentum), _affine(affine), _track_running_stats(track_running_stats)
    {
        if (affine)
        {
            _alpha = create_param("alpha", last_dims, TensorInit_Types::One);
            _beta = create_param("beta", last_dims, TensorInit_Types::Zero);
        }

        if (track_running_stats)
        {
            _num_batches_tracked = 0;
            _running_mean.reset(last_dims, TensorInit_Types::Zero);
            _running_var.reset(last_dims, TensorInit_Types::One);
        }
    }

    // input: x[], each last_size calc one mean/var
    // x[i] = (x[i] - mean) / sqrt(variance + e) * alpha + beta;
    // Note: elems in one last_size don't share alpha/beta, the same elem across different groups share same alpha/beta
    OVERRIDE void forward(const Tensor &x, Tensor &y) const
    {
        assert(x.shape() > 0);
        assert(x.dim().match_bottom(_last_dims));
        uint last_dims = _last_dims.size();
        uint first_match_dims = x.shape() - last_dims;

        // Note: mean/var will affect x's grad
        auto mean = x.avg(last_dims), var = x.var(false, last_dims);

        y = x.add(mean, 1.0, -1.0, 0, first_match_dims, 0).mul_(var.pow(-0.5, EPSILON), 1, 0, first_match_dims, 0);
        if (_affine)
        {
            y.mul_(_alpha, 1, 0, 0, last_dims).add_(_beta, 1, 1, 0, 0, last_dims);
        }

        double exp_avg_factor = _momentum;
        if (_is_train && _track_running_stats)
        {
            _num_batches_tracked++;
            if (_momentum == 0)
            {
                exp_avg_factor = 1.0 / _num_batches_tracked;
            }

            _running_mean.add(mean.data(), _running_mean, exp_avg_factor, 1 - exp_avg_factor);
            _running_var.add(var.data(), _running_var, exp_avg_factor, 1 - exp_avg_factor);
        }
    }
};