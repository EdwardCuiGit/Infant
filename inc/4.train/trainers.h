#pragma once

#include "inc/4.train/raw_trainer.h"

#include "inc/3.network/sample_cnn.h"
#include "inc/3.network/transformer.h"
#include <filesystem>

class SampleCnnTrainer: public RawTrainer
{
public:
    bool train(const Tensor &x, const Tensor &z, TrainerConfig trainer_config) const
    {
        trainer_config.is_print_functor = true;
        Ptr<Operator> network = std::make_shared<SampleCnn>();
        InMemoryDataLoader x_loader(x);
        InMemoryDataLoader z_loader(z);
        return core_train(x_loader, z_loader, network, trainer_config);
    }
};

class TransformerTrainer : public RawTrainer
{
public:
    bool train()
    {
        uint hidden_dim = 6;
        uint max_node_count = 9;
        TrainerConfig config;
        config.iterations = 200;
        config.mini_batch = 200;
        config.is_print_functor = false;
        config.optim_config.learning_rate = 0.1;
        config.training_data_file_path = "C:/Code/EdwardCodeWorkSpace/Infant/unit_tests/arithmetic_input.txt";
        config.label_data_file_path= "C:/Code/EdwardCodeWorkSpace/Infant/unit_tests/arithmetic_output.txt";
        config.model_save_path = "C:/Code/EdwardCodeWorkSpace/Infant/unit_tests/transformer_model.txt";

        auto curr_dir = std::filesystem::current_path();

        TextFileLoader x_loader(config.training_data_file_path, max_node_count);
        TextFileLoader z_loader(config.label_data_file_path, max_node_count);

        Transformer::Config network_c;
        network_c.decoder_only() = true;
        network_c.num_decoder_layers() = 2;
        network_c.has_embedding() = true;
        network_c.dict_size() = 20;
        network_c.has_position_embedding() = true;
        network_c.has_padding_mask() = true;

        network_c.decoder_sa().hidden_dim() = hidden_dim;
        network_c.decoder_sa().node_len() = max_node_count;
        network_c.decoder_sa().multi_head() = 2;
        network_c.decoder_sa().dk() = 1.0;
        network_c.decoder_sa().fc_intermediate_factor() = 4;
        network_c.decoder_sa().has_bias() = false;
        network_c.decoder_sa().init_type() = (uint)TensorInit_Types::Gaussian;
        network_c.decoder_sa().bias_init_type() = (uint)TensorInit_Types::Zero;

        Ptr<Operator> network = std::make_shared<Transformer>(network_c);
        return core_train(x_loader, z_loader, network, config);
    }

    bool inference()
    {
        return true;
    }
};