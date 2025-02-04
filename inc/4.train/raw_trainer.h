#pragma once

#include "../4.train/data_loader.h"
#include "optimizer_base.h"
#include "../1.functors/functor_graph.h"

struct TrainerConfig
{
public:
    std::string training_data_file_path;
    std::string label_data_file_path;
    std::string model_save_path;
    uint mini_batch = 2;
    uint iterations = 100;
    bool is_dynamic = false;
    bool is_print_functor = false;
    OptimizerConfig optim_config;
};

class RawTrainer
{
public:
    RawTrainer()
    {
        Environment::Init();
    }

    bool core_train(DataLoader& x_loader, DataLoader& z_loader, Ptr<Operator> network, TrainerConfig trainer_config) const
    {
        Environment::Set_Train();
        Environment::Set_Print_Functor(trainer_config.is_print_functor);
        Ptr<Optimizer> optim = Optimizer::Create(trainer_config.optim_config);
        if (optim == nullptr)
        {
            LOG_ERROR("optim create failed");
            return false;
        }

        // TODO: only supports one loss
        bool first_iteration = true;
        uint eporch = 0;
        uint total_samples = 0;
        uint samples_per_eporch = 0;
        Vector<double> losses;
        FunctorGraph::singleton().reset();
        TensorList ys;
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
                // this is one mini batch
                // all the features are filled;
                if (trainer_config.is_dynamic || first_iteration || !network->is_const())
                {
                    TensorList xs = {input_x, input_z};
                    FunctorGraph::singleton().set_inputs(xs);

                    ys = network->forward(xs);

                    FunctorGraph::singleton().set_outputs({ys[0]});
                    FunctorGraph::singleton().print(std::cout);
                }
                else
                {
                    ys = FunctorGraph::singleton().forward({input_x, input_z});
                }

                Tensor loss = ys[0];
                FunctorGraph::singleton().stop_auto_grad();
                double loss_value = loss.avg().data().first_item();
                FunctorGraph::singleton().start_auto_grad();
                losses.push_back(loss_value);

                // all the grads for features & params are filled;
                loss.grad().reset(loss.dim(), TensorInit_Types::One);
                FunctorGraph::singleton().backward({loss});
                FunctorGraph::singleton().zero_features();

                optim->step(FunctorGraph::singleton().params());
                FunctorGraph::singleton().zero_grads();

                if (trainer_config.is_dynamic || !network->is_const())
                {
                    FunctorGraph::singleton().reset();
                }

                LOG_INFO("eporch: " << eporch << " ,iter: " << i << " ,loss: " << loss_value << " , lr: " << optim->learning_rate());
                if (eporch == 0)
                    samples_per_eporch += samples_x;
                total_samples += samples_x;

                first_iteration = false;
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
        LOG_INFO("Final loss:" << losses[losses.size() - 1]);

        if (!trainer_config.model_save_path.empty())
        {
            std::ofstream ofile(trainer_config.model_save_path);
            if (!ofile)
            {
                LOG_ERROR("error open file\n");
                return false;
            }

            network->save_op(ofile);
            ofile.close();
        }


        Environment::Set_Train(false);
        return true;
    }

};
