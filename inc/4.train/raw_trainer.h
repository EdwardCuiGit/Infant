#pragma once

#include "../3.network/network.h"
#include "../3.network/data_loader.h"
#include "optimizer_base.h"

struct TrainerConfig
{
public:
    uint mini_batch;
    uint iterations;
};

class RawTrainer
{
public:
    virtual bool train(const Tensor& x, const Tensor& z, Network& network, OptimizerConfig optim_config, TrainerConfig trainer_config, Vector<double> &losses) const
    {
        Ptr<Optimizer> optim = Optimizer::create(optim_config);
        if (optim == nullptr)
        {
            LOG_ERROR("optim create failed");
            return false;
        }

        InMemoryDataLoader x_loader(x);
        InMemoryDataLoader z_loader(z);

        //TODO: only supports one loss
        bool first_iteration = true;
        uint eporch = 0;
        uint total_samples = 0;
        uint samples_per_eporch = 0;
        network.reset(); // this is used to cleanup nodes in graph first
        for (uint i = 0; i < trainer_config.iterations; ++i)
        {
            // one itertion will read and process mini_batch number of samples
            // last iteration will have not enough samples
            // note: only supports one data loader first
            // note: only supports one input & one output first
            Tensor input_x, input_z, output_y;
            uint samples_x = x_loader.read_next(input_x, trainer_config.mini_batch);
            uint samples_z = z_loader.read_next(input_z, trainer_config.mini_batch);
            assert(samples_x == samples_z);
            if (samples_x > 0)
            {
                if (first_iteration)
                {
                    auto ys = network.build_network({input_x, input_z});
                    output_y = ys[0];
                    network.set_train(true);
                    first_iteration = false;
                }
                else
                {
                    // this is one mini batch
                    // all the features are filled;
                    auto ys = network.forward({input_x, input_z});
                    output_y = ys[0];
                }

                double loss = network.calc_loss(output_y);
                losses.push_back(loss);

                // all the grads for features & params are filled;
                output_y.grad().reset(output_y.dim(), TensorInit_Types::One);
                network.backward({output_y});
                network.zero_features();

                optim->step(network.params());
                network.zero_grads();


                LOG_INFO("eporch: " << eporch << " ,iter: " << i << " ,loss: " << loss << " , lr: " << optim->learning_rate());
                if (eporch == 0)
                    samples_per_eporch += samples_x;
                total_samples += samples_x;
            }
            else
            {
                x_loader.reset();
                z_loader.reset();
                eporch += 1;
                --i;
            }
        }

        LOG_INFO("\nsamples_per_eporch:" << samples_per_eporch);
        LOG_INFO("total samples processed(count each eporch):" << total_samples);

        return true;
    }
};