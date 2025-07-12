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
public:
    NormBase(ConfigBase* ptr, const std::string& type) : UnOp(ptr, type){}
    // as there are mutable variables
    virtual bool is_const() const override
    {
        return false;
    }
};

// not saving save/load
class BatchNorm2d : public NormBase
{
private:
    Tensor _alpha, _beta;
    mutable TensorD<float> _running_mean, _running_var; // assigned but not used
    mutable uint _num_batches_tracked;
    uint _channels;
    float _momentum;
    bool _affine;
    bool _track_running_stats;

public:
    BatchNorm2d(uint channels, float momentum = 0.1, bool affine = true, bool track_running_stats = true)
        : NormBase(nullptr, "BatchNorm2d"), _channels(channels), _momentum(momentum), _affine(affine), _track_running_stats(track_running_stats)
    {
        assert(channels > 0);
        if (affine)
        {
            _alpha = add_param("alpha", {_channels}, TensorInit_Types::One);
            _beta = add_param("beta", {_channels}, TensorInit_Types::Zero);
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
    virtual Tensor forward(const Tensor &x) const override
    {
        assert(x.shape() >= 4);
        assert(x.dim()[1] == _channels);
        auto x_c = x.swap(0, 1);                // [C, N, H, W]
        auto mean = x_c.avg(3), var = x_c.var(false, 3); // mean/var: [C]

        // note: perf: change to one loop
        x_c.add_(mean, 1, -1, 0, 1, 0).mul_(var.pow(-0.5, EPSILON), 1, 0, 1, 0);
        if (_affine)
        {
            x_c.mul_(_alpha, 1, 0, 1, 0).add_(_beta, 1, 1, 0, 1, 0);
        }

        Tensor y = x_c.swap(0, 1);

        float exp_avg_factor = _momentum;
        if (Environment::Is_Train() && _track_running_stats)
        {
            _num_batches_tracked++;
            if (_momentum == 0)
            {
                exp_avg_factor = 1.0 / _num_batches_tracked;
            }

            _running_mean.add(mean.data(), _running_mean, exp_avg_factor, 1 - exp_avg_factor);
            _running_var.add(var.data(), _running_var, exp_avg_factor, 1 - exp_avg_factor);
        }

        return y;
    }
};

class LayerNorm : public NormBase
{
public:
    // this is to support auto save/load
    struct Config : ConfigBase
    {
        DEFINE_FIELD(bool, has_lm, false);

        Vector<uint> &last_dims() { return access_uint_vector("last_dims"); } 
        const Vector<uint> last_dims() const { return access_uint_vector("last_dims"); } 
        // const Vector<uint> last_dims; // note: we can't use T& as class members

        DEFINE_FIELD(float, momentum, 0.1);
        DEFINE_FIELD(bool, affine, true);
        DEFINE_FIELD(bool, track_running_stats, false);

        Config(const Vector<uint>& last_dims = {}, bool has_lm = false, float momentum = 0.1, bool affine = true, bool track_running_stats = false)
        : ConfigBase("LayerNorm")
        {
            this->has_lm() = has_lm;
            this->last_dims() = last_dims;
            this->momentum() = momentum;
            this->affine() = affine;
            this->track_running_stats() = track_running_stats;
        }
    };

private:
    Config _c;
    Tensor _alpha, _beta;
    // TODO: _running_mean & _running_var & _num_batches_tracked only used in first_forward, and we need to keep this LayerNorm in the graph
    mutable TensorD<float> _running_mean, _running_var; // assigned but not used
    mutable uint _num_batches_tracked;
public:
    // LayerNorm() : NormBase(&_c, "LayerNorm"){}
    LayerNorm(const Config& c) : NormBase(&_c, "LayerNorm"), _c(c)
    {
        if (!c.has_lm())
            return;
        if (c.affine())
        {
            _alpha = add_param("alpha", c.last_dims(), TensorInit_Types::One);
            _beta = add_param("beta", c.last_dims(), TensorInit_Types::Zero);
        }

        if (c.track_running_stats())
        {
            _num_batches_tracked = 0;
            _running_mean.reset(c.last_dims(), TensorInit_Types::Zero);
            _running_var.reset(c.last_dims(), TensorInit_Types::One);
        }
    }

    // input: x[], each last_size calc one mean/var
    // x[i] = (x[i] - mean) / sqrt(variance + e) * alpha + beta;
    // Note: elems in one last_size don't share alpha/beta, the same elem across different groups share same alpha/beta
    virtual Tensor forward(const Tensor &x) const override
    {
        if (!_c.has_lm())
        {
            return x;
        }

        assert(x.shape() > 0);
        assert(x.dim().match_bottom(_c.last_dims()));
        uint last_dims = _c.last_dims().size();
        uint first_match_dims = x.shape() - last_dims;

        // Note: mean/var will affect x's grad
        auto mean = x.avg(last_dims), var = x.var(false, last_dims);

        Tensor y = x.add(mean, 1.0, -1.0, 0, first_match_dims, 0).mul_(var.pow(-0.5, EPSILON), 1, 0, first_match_dims, 0);
        if (_c.affine())
        {
            y.mul_(_alpha, 1, 0, 0, last_dims).add_(_beta, 1, 1, 0, 0, last_dims);
        }

        float exp_avg_factor = _c.momentum();
        if (Environment::Is_Train() && _c.track_running_stats())
        {
            _num_batches_tracked++;
            if (_c.momentum() == 0)
            {
                exp_avg_factor = 1.0 / _num_batches_tracked;
            }

            _running_mean.add(mean.data(), _running_mean, exp_avg_factor, 1 - exp_avg_factor);
            _running_var.add(var.data(), _running_var, exp_avg_factor, 1 - exp_avg_factor);
        }

        return y;
    }

    virtual bool is_const() const
    {
        return !(_c.has_lm() && Environment::Is_Train() && _c.track_running_stats());
    }
};

class RMSNorm : public NormBase
{
public:
    struct Config : ConfigBase
    {
        DEFINE_FIELD(bool, enabled, false);
        DEFINE_FIELD(float, gamma, 1.0f);

        Config(bool enabled = false, float gamma = 1.0f)
        : ConfigBase("RMSNorm")
        {
            this->enabled() = enabled;
            this->gamma() = gamma;
        }
    };

private:
    Config _c;

public:
    RMSNorm(const Config& c) : NormBase(&_c, "RMSNorm"), _c(c)
    {
    }

    virtual Tensor forward(const Tensor &x) const override
    {
        if (!_c.enabled())
            return x;
        return x.rms_norm(_c.gamma());
    }
};
