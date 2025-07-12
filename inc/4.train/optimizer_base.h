#pragma once

#include "inc/2.operators/operator_base.h"

enum class Optimizer_Types : uint
{
    Sgd,
    Adam,
};

struct OptimizerConfig
{
public:
    Optimizer_Types optim_type = Optimizer_Types::Sgd;
    float learning_rate = 0.000001;
    float weight_decay = 0;
    float momentum = 0.2;
};

// How to add new operator: inherit Optimizer, add type to Optimizer_Types, add one row in create(), all in this file
// TODO: OptimizerConfig for all hyper parameters
// TODO: implement all major optimizers
// TODO: weight decays;
// TODO: parameter groups
// TODO: make each optimizer a separate file
class Optimizer
{
protected:
    OptimizerConfig _config;

public:
    static bool Parse_config(const std::string &config_file, OptimizerConfig &config);
    static Ptr<Optimizer> Create(const std::string &config_file);
    static Ptr<Optimizer> Create(const OptimizerConfig &config);

    Optimizer()
    {
    }

    virtual ~Optimizer()
    {
    }

    Optimizer(const OptimizerConfig &config)
    {
        _config = config;
    }

    // reset values to be initial value
    virtual void init()
    {
        _config.learning_rate = _config.learning_rate; // TODO, useless now
    }

    // during training, lr will be changed
    virtual void step(TensorList &params) const = 0;
    float learning_rate() const
    {
        return _config.learning_rate;
    }
};

class Sgd : public Optimizer
{
public:
    Sgd(float lr, float momentum = 0, float weight_decay = 0) : Optimizer()
    {
        assert(lr >= 0);
        assert(momentum >= 0);
        assert(weight_decay >= 0);
        _config.learning_rate = lr;
        _config.momentum = momentum;
        _config.weight_decay = weight_decay;
    }

    Sgd(const OptimizerConfig& config): Optimizer()
    {
        _config = config;
    }

    // TODO: not finished yet for parameter group
    virtual void step(TensorList &params) const override
    {
        // TODO: implement momentum, etc.;
        for (Tensor param : params)
        {
            if (_config.weight_decay > 0)
            {
                // TODO
            }

            TensorDP grad;
            if (_config.momentum > 0)
            {
                if (param.momentum().size() == 0)
                {
                    param.momentum().reset(param.dim(), TensorInit_Types::Zero);
                }

                TensorD<float> grad_sum;
                param.grad().sum(grad_sum);
                if (grad_sum.first_item() == 0) // address gradient vanishing problem
                {
                    param.grad().reset(param.dim(),  TensorInit_Types::Gaussian);
                }

                // NG's formula:
                //  grad = param->momentum->add_(param->grad(), 0, -1, _config.momentum, 1 - _config.momentum);
                //  param->add_(grad, 1.0, -1.0 * _config.learning_rate);
                //  typical formula
                TensorD<float> new_momentum, new_data;
                param.momentum().add(param.grad(), new_momentum, _config.momentum, 1, 0);
                param.momentum().deep_copy(new_momentum);
                param.data().add(param.momentum(), new_data, 1, -1 * _config.learning_rate, 0);
                param.data().deep_copy(new_data);
            }
            else
            {
                TensorD<float> new_data;
                param.data().add(param.grad(), new_data, 1, -1 * _config.learning_rate, 0);
                param.data().deep_copy(new_data);
            }
        }
    }
};

bool Optimizer::Parse_config(const std::string &config_file, OptimizerConfig &config)
{
    // TODO
    assert(false);
    return false;
}

Ptr<Optimizer> Optimizer::Create(const std::string &config_file)
{
    OptimizerConfig config;
    bool ok = Parse_config(config_file, config);
    if (!ok)
    {
        LOG_ERROR("parse optimizer config failed\n"
                  << 1);
        return nullptr;
    }

    return Create(config);
}

Ptr<Optimizer> Optimizer::Create(const OptimizerConfig &config)
{
    /*std::map<Optimizer_Types, Ptr<Optimizer>> instances = {
        {Optimizer_Types::SGD, std::make_shared<Sgd>()},
        {Optimizer_Types::ADAM, std::make_shared<Adam>()},
    };*/

    switch (config.optim_type)
    {
    case Optimizer_Types::Sgd:
        return std::make_shared<Sgd>(config);
    default:
        assert(false);
    }

    return nullptr;
}