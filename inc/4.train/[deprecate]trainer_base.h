#ifdef __UNDEFINED__
#pragma once

#include "3.network/data_loader.h"
#include "3.network/network.h"
#include "optimizer_base.h"

struct TrainerConfig
{
public:
    uint mini_batch;
    uint iterations;
};

class Trainer
{
private:
    static bool parse_config(const std::string &training_config_file, TrainerConfig &config)
    {
        // TODO
        return false;
    }

public:
    // can't use now
    // configuration files as input
    virtual bool train(const std::string &network_config_file, const std::string &training_config_file) const
    {
        Network network;
        if (!network.init(network_config_file))
        {
            LOG_ERROR("network init failed");
            return false;
        }

        Ptr<Optimizer> optim = Optimizer::create(training_config_file);
        if (optim == nullptr)
        {
            LOG_ERROR("optim create failed");
            return false;
        }

        DataLoadersManager data_loaders_manager;
        if (!data_loaders_manager.init(training_config_file))
        {
            LOG_ERROR("data loaders manager init failed");
            return false;
        }

        TrainerConfig trainer_config;
        if (!parse_config(training_config_file, trainer_config))
        {
            LOG_ERROR("trainer config parse failed");
            return false;
        }

        return this->train(network, *optim.get(), data_loaders_manager, trainer_config);
    }

    // configuration objects based
    virtual bool train(NetworkGraph &graph, const OptimizerConfig &optim_config, const DataLoadersManagerConfig &dlm_config,
                       const TrainerConfig &trainer_config) const
    {
        Network network;
        if (!network.init(graph))
        {
            LOG_ERROR("network init failed");
            return false;
        }

        Ptr<Optimizer> optim = Optimizer::create(optim_config);
        if (optim == nullptr)
        {
            LOG_ERROR("optim create failed");
            return false;
        }

        DataLoadersManager data_loaders_manager;
        if (!data_loaders_manager.init(dlm_config))
        {
            LOG_ERROR("data loaders manager init failed");
            return false;
        }

        this->train(network, *optim.get(), data_loaders_manager, trainer_config);
    }

    virtual bool train(Network &network, Optimizer &optim, DataLoadersManager &data_loaders_manager,
                       const TrainerConfig &config) const
    {
        optim.reset();
        network.set_train(true);

        //TODO: only supports one data loader first
        //TODO: only supports one input & one output first
        //TODO: only supports one loss
        TensorPtr start_x;
        TensorPtr end_y;
        TensorPtr end_x, end_y_grad, start_x_grad; // place holders only

        uint eporch = 0;
        uint total_samples = 0;
        uint samples_per_eporch = 0;
        for (uint i = 0; i < config.iterations; ++i)
        {
            // one itertion will read and process mini_batch number of samples
            // last iteration will have not enough samples
            uint samples = data_loaders_manager[0]->read_next(start_x, config.mini_batch);
            if (samples > 0)
            {
                // this is one mini batch
                // all the features are filled;
                network.forward(start_x, end_y);
                double loss = 0.0; //end_y[0]; //TODO: mock code

                // all the grads for features & params are filled;
                network.backward(end_x, end_y, end_y_grad, start_x_grad);
                network.zero_feature();

                optim.step(network.params());
                network.zero_grad();

                LOG_INFO("eporch: " << eporch << " ,iter: " << i << " ,loss: " << loss << " , lr: " << optim.learning_rate());

                if (eporch == 0)
                    samples_per_eporch += samples;
                total_samples += samples;
            }
            else
            {
                data_loaders_manager[0]->reset();
                eporch += 1;
            }
        }

        LOG_INFO("samples_per_eporch:" << samples_per_eporch);
        LOG_INFO("total samples processed(count each eporch):" << total_samples);

        network.set_train(false);
        return true;
    }
};
#endif