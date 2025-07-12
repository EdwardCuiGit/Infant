#pragma once

#include "inc/2.operators/operator_base.h"
#include "inc/2.operators/attentions.h"
#include "inc/2.operators/norm.h"
#include "inc/2.operators/fc.h"

// below supports encoder_decoder and decoder_only
// variant projection op: tensor<N, L, H> -> tensor<N, L', H>
class Transformer : public Operator
{
    friend class TestTransformer;

public:
    struct Config : ConfigBase
    {
        DEFINE_FIELD(bool, decoder_only, true);
        DEFINE_FIELD(uint, num_encoder_layers, 1);
        DEFINE_FIELD(uint, num_decoder_layers, 1);
        DEFINE_FIELD(bool, has_embedding, false);
        DEFINE_FIELD(uint, dict_size, 1);
        DEFINE_FIELD(bool, has_position_embedding, false);
        DEFINE_FIELD(bool, has_padding_mask, false);

        // below are core attention configs, instead of using Attention::Config, use raw ones
        DEFINE_SUB_CONFIG(Attention, decoder_sa);

        // no need to assign is_cross_attention, triange_mask, single_node, has_fc
        Config(bool decoder_only = true, uint num_encoder_layers = 1, uint num_decoder_layers = 1,
               bool has_embedding = false, uint dict_size = 1,
               bool has_position_embedding = false,
               bool has_padding_mask = false,
               uint hidden_dim = 1, uint node_len = 1, uint multi_head = 1, float dk = 1.0,
               uint fc_intermediate_factor = 4,
               bool has_moe = false,
               uint total_experts = 2, uint active_experts = 1,
               bool has_mla = false,
               bool has_bias = false, TensorInit_Types init_type = TensorInit_Types::Gaussian, TensorInit_Types bias_init_type = TensorInit_Types::Zero,
               const LayerNorm::Config &lm = LayerNorm::Config()) : ConfigBase("Transformer")
        {
            this->decoder_only() = decoder_only;
            this->num_encoder_layers() = num_decoder_layers;
            this->num_decoder_layers() = num_encoder_layers;
            this->has_embedding() = has_embedding;
            this->dict_size() = dict_size;
            this->has_position_embedding() = has_position_embedding;
            this->has_padding_mask() = has_padding_mask;
            this->decoder_sa() = Attention::Config(hidden_dim, node_len, multi_head, false, dk, false,
                                                   fc_intermediate_factor, has_moe, total_experts, active_experts, has_mla, has_bias, init_type, bias_init_type, lm);
        }
    };

private:
    Config _c;

    Array<Ptr<Attention>> _encoder_self_attentions;
    Array<Ptr<Attention>> _decoder_self_attentions;
    Array<Ptr<Attention>> _decoder_cross_attentions;
    Tensor _embedding;
    Tensor _position_embedding;

public:
    Transformer(const Config &c) : _c(c), Operator(&_c, "Transformer")
    {
        if (!_c.decoder_only())
        {
            assert(_c.num_encoder_layers() > 0);
            _encoder_self_attentions.reserve(_c.num_encoder_layers());
            for (uint i = 0; i < _c.num_encoder_layers(); ++i)
            {
                Attention::Config encoder_sa_c(_c.decoder_sa()); // copy from decoder_sa config
                encoder_sa_c.is_cross_attention() = false;
                encoder_sa_c.has_fc() = true;
                _encoder_self_attentions[i] = add_op<Attention>(StringUtil::concat("esa", i), encoder_sa_c);
            }

            _c.decoder_sa().has_fc() = false;
        }
        else
        {
            // decoder_only mode, has fc in encoding stage, no fc in decoding stage
            _c.decoder_sa().has_fc() = true;
        }

        _decoder_self_attentions.reserve(_c.num_decoder_layers());
        _decoder_cross_attentions.reserve(_c.num_decoder_layers());
        _c.decoder_sa().is_cross_attention() = false;

        for (uint i = 0; i < _c.num_decoder_layers(); ++i)
        {
            _decoder_self_attentions[i] = add_op<Attention>(StringUtil::concat("dsa", i), _c.decoder_sa());

            Attention::Config decoder_ca_c(_c.decoder_sa());
            decoder_ca_c.is_cross_attention() = true;
            decoder_ca_c.has_fc() = true;
            _decoder_cross_attentions[i] = add_op<Attention>(StringUtil::concat("dca", i), decoder_ca_c);
        }

        if (_c.has_embedding())
        {
            _embedding = add_param("embedding", {_c.dict_size(), _c.decoder_sa().hidden_dim()},
                                   (TensorInit_Types)_c.decoder_sa().init_type());
        }

        if (_c.has_position_embedding())
        {
            _position_embedding = add_param("position_embedding", {_c.decoder_sa().node_len(), _c.decoder_sa().hidden_dim()},
                                            (TensorInit_Types)_c.decoder_sa().init_type());
        }
    }

    /*
    special words:
    dict_id == 0: <pad>
    dict_id == 1: <start>
    dict_id == 2: <end>

    for train: xs[0] is input sequence, xs[1] is target sequence
    x[0]: [batch_size, node_len], each float represent one word id
    x[1]: [batch_size, node_len], each float represent one word id
    x[0] is like 'what's the capital of China?
    x[1] is similar, represents target text sequence
    y[0]:  single value, loss
    for inference: xs[0] is input sequence
    x[0]: [batch_size, node_len], each float represent one word id
    y[0]: [batch_size, node_len], each float represent one word id
    */
    virtual TensorList forward(const TensorList &xs) const override
    {
        assert(xs.size() >= 1);
        Tensor x0 = xs[0];
        Tensor x0_padding_mask;
        if (_c.has_embedding())
        {
            // TODO: below encode, decode methods should join the graph, so that grad can pass to input_embedding and output_projection?
            if (_c.has_padding_mask())
            {
                x0_padding_mask = x0.replace(0.0f, 0.0f, 1.0f);
            }

            x0 = x0.encode_by_dict(_embedding); // for last dim's each id, find in input_embedding, and build a new one
            if (_c.has_position_embedding())
            {
                // [batch_size, node_len, hidden_dim].add([node_len, hidden_dim])
                x0 = x0.add(_position_embedding, 1, 1, 0, 0, 2);
            }
        }

        TensorList ys;
        if (Environment::Is_Train())
        {
            assert(xs.size() >= 2);
            Tensor x1 = xs[1];
            Tensor x1_padding_mask;
            if (_c.has_embedding())
            {
                x1 = x1.insert(0, 1, 1);

                if (_c.has_padding_mask())
                {
                    x1_padding_mask = x1.replace(0.0f, 0.0f, 1.0f);
                }

                x1 = x1.encode_by_dict(_embedding);
                if (_c.has_position_embedding())
                {
                    x1 = x1.add(_position_embedding, 1, 1, 0, 0, 2);
                }
            }

            // {batch_size, node_len, hidden_dim}
            ys = _forward_train({x0, x1, x0_padding_mask, x1_padding_mask});
            if (_c.has_embedding())
            {
                ys = _project_to_vocab(ys);
                // get each target word's probability by encode()
                // note: 0 below is padding_id, which is to skip padding's grad
                Tensor tp = xs[1].search_by_dict(ys[1], 0); // {batch_size, node_len}.encode{batch_size, node_len, dict_size}
                // => batch_size, node_len
                Tensor loss = tp.ln(0.0001).linear(-1).avg(1); // {batch_size}, 
                ys = {loss}; // note: final output shall be the same as last functor's output
            }
            else // note: only used for unit test
            {
                // loss: 1 - cos(output[i], target[i])
                // Tensor loss = ys[0].dot(xs[1], 2).avg(1).linear_(-1, 1);
                // ys = {loss};
                ys = {ys[0]};
            }
        }
        else
        {
            ys = _forward_inference({x0, x0_padding_mask});

            if (_c.has_embedding())
            {
                ys = _project_to_vocab(ys);
                ys = {ys[0]};
            }
        }

        return ys;
    }

private:
    TensorList _project_to_vocab(const TensorList &ys) const
    {
        assert(ys.size() == 1);
        Tensor y = ys[0];              // {batch_size, node_len, hidden_dim}
        Tensor y1 = y.dot(_embedding); // {batch_size, node_len, dict_size}
        y1.softmax_(1);                // {batch_size, node_len, dict_size}
        // note: what if decode generates padding, start? keep it as it is
        Tensor output = y1.decode__(); // {batch_size, node_len} // output sequences
        return {output, y1};
    }

    TensorList _encode(const Tensor &x, const Tensor &x_padding_mask) const
    {
        TensorList ys;
        Tensor x_encoder = x;
        if (_c.decoder_only())
        {
            // encoding every input token in forward direction
            // very similar to encoder, just that it's forward direction only
            for (uint i = 0; i < _decoder_self_attentions.size(); ++i)
            {
                // masked self attention + add + norm
                ys = _decoder_self_attentions[i]->forward({x_encoder, x_padding_mask});
                // => [batch_size, input_node_len, hidden_dim]

                x_encoder = ys[0];
            }
        }
        else
        {
            for (uint i = 0; i < _encoder_self_attentions.size(); ++i)
            {
                // masked self attention + add + norm
                ys = _encoder_self_attentions[i]->forward({x_encoder, x_padding_mask});
                // => [batch_size, input_node_len, hidden_dim]

                x_encoder = ys[0];
            }
        }

        return ys;
    }

    /*
    inputs are input sequence's encoding, and target output sequence's encoding
    outputs are each target output sequence's encoding
    input: xs:
        x_encoder[batch_size, node_len_encoder, hidden_dim]
        x_decoder[batch_size, node_len_decoder, hidden_dim], is the whole node sequence with fixed length after padding
    output: ys:
        y[batch_size, node_len_decoder, hidden_dim]
    */
    TensorList _forward_train(const TensorList &xs) const
    {
        assert(xs.size() >= 2);
        Tensor x_encoder = xs[0];
        Tensor x_decoder = xs[1];
        Tensor x_encoder_padding_mask;
        Tensor x_decoder_padding_mask;
        if (_c.has_padding_mask())
        {
            assert(xs.size() >= 4);
            x_encoder_padding_mask = xs[2];
            x_decoder_padding_mask = xs[3];
        }

        // step-1: encoding
        TensorList ys_encoder = _encode(x_encoder, x_encoder_padding_mask);

        // step-2: decoding
        uint decoder_layer_size = _decoder_self_attentions.size();
        Tensor y_sa;
        for (uint i = 0; i < decoder_layer_size; ++i)
        {
            // masked self attention + add + norm
            TensorList ys_sa = _decoder_self_attentions[i]->forward({x_decoder, x_decoder_padding_mask}, true, false);

            // cross attention + add + norm + fc
            TensorList ys_ca = _decoder_cross_attentions[i]->forward({ys_sa[0], ys_sa[1], ys_encoder[2], ys_encoder[3],
                                                                     x_decoder_padding_mask, x_encoder_padding_mask});

            x_decoder = ys_ca[0];
        }

        return {x_decoder};
    }

    /*
    inputs are input sequence's k/v cache, outputs are a new sequence of token's encoding
    input: xs:
        x_encoder[batch_size, node_len_encoder, input_dim]
    output: ys:
        y[batch_size, node_len_decoder, output_dim]
    */

    Tensor _get_start_node() const
    {
        Tensor start_node;
        start_node.reset({1}, TensorInit_Types::One);
        if (_c.has_embedding())
        {
            start_node = start_node.encode_by_dict(_embedding);

            if (_c.has_position_embedding())
            {
                start_node = start_node.add(_position_embedding.subset({_c.decoder_sa().hidden_dim()}, 0));
            }
        }
        else
        {
            start_node.reset({1, _c.decoder_sa().hidden_dim()}, TensorInit_Types::Ordinal);
        }

        return start_node;
    }

    Tensor _get_padding_node() const
    {
        Tensor padding_node;
        padding_node.reset({1}, TensorInit_Types::Zero);
        if (_c.has_embedding())
        {
            padding_node = padding_node.encode_by_dict(_embedding).merge_dim(0, 2);
        }
        else
        {
            padding_node.reset({_c.decoder_sa().hidden_dim()}, TensorInit_Types::Zero);
        }
        return padding_node;
    }

    Tensor _get_end_node() const
    {
        Tensor end_node;
        end_node.reset({1}, {2});
        if (_c.has_embedding())
        {
            end_node = end_node.encode_by_dict(_embedding).merge_dim(0, 2);
        }
        else
        {
            end_node.reset({_c.decoder_sa().hidden_dim()}, TensorInit_Types::Ordinal);
        }
        return end_node;
    }

    /* note: better perf way is to have streaming input tensors, after one sequence finished processing, one more just filled and replace
    its position
    xs: x[batch_size, node_len, hidden_dim]
    ys: [batch_size, node_len_decoder, hidden_dim]
       ..
    */
    TensorList _forward_inference(const TensorList &xs) const
    {
        assert(xs.size() >= 1);
        Tensor x_encoder = xs[0];
        Tensor x_encoder_padding_mask;
        if (_c.has_padding_mask())
        {
            assert(xs.size() >= 2);
            x_encoder_padding_mask = xs[1];
        }

        uint batch_size = x_encoder.dim()[0];
        Array<TensorList> ys(batch_size);
        uint decoder_layer_size = _decoder_self_attentions.size();

        // step-1: encoding, same as training
        TensorList ys_encoder = _encode(x_encoder, x_encoder_padding_mask);

        // step-2: decoding, needs to do one by one
        // [batch_size, 1, hidden_dim]
        Tensor start_node = _get_start_node();
        Tensor end_node = _get_end_node();
        Tensor curr_nodes;
        // note: start node set to be ordinal, as it's the first node
        for (uint i = 0; i < batch_size; ++i)
            curr_nodes.append_(start_node);

        Tensor curr_nodes_padding_mask; // no mask for curr_node;
        if (_c.has_padding_mask())
        {
            curr_nodes_padding_mask.reset(curr_nodes.dim(), TensorInit_Types::One);
        }

        // note: we used k_cache, v_cache to store k/v values from previous node, as more node generated, these 2 cache tensors keep growing
        // we shall pre-allocate memory to improve perf
        TensorList k_cache(decoder_layer_size); // [decoder_layer_size, batch_size, multi_head, curr_node_len, output_dim]
        TensorList v_cache(decoder_layer_size); // [decoder_layer_size, batch_size, multi_head, output_dim, curr_node_len]

        // this loop will do one more node generation each time, until all the node finished
        uint num_completed = 0;
        uint curr_node_len = 0;
        Array<bool> completed(batch_size);
        while (num_completed < batch_size && curr_node_len < _c.decoder_sa().node_len())
        {
            for (uint layer = 0; layer < decoder_layer_size; ++layer)
            {
                // step-1: self-attentions
                // [batch_size, input_dim] * [multi_head, output_dim, input_dim] => [batch_size, multi_head, output_dim]

                TensorList ys;
                if (!_c.decoder_sa().has_mla())
                {
                    // this will return both new y, q, and also update k/v_cache
                    ys = _decoder_self_attentions[layer]->forward({curr_nodes, k_cache[layer], v_cache[layer]}, false, false, true);
                    k_cache[layer] = ys[2];
                    v_cache[layer] = ys[3];
                }
                else
                {
                    ys = _decoder_self_attentions[layer]->forward({curr_nodes, k_cache[layer]}, false, false, true);
                    k_cache[layer] = ys[2];
                }


                // step-2: cross_attention + fc
                // note: this must be single node mode execution
                TensorList ys_ca = _decoder_cross_attentions[layer]->forward(
                    {ys[0], ys[1], ys_encoder[2], ys_encoder[3], curr_nodes_padding_mask, x_encoder_padding_mask}, false, true, true);

                // step-4: move to next layer
                curr_nodes = ys_ca[0];
            }

            for (uint i = 0; i < batch_size; ++i)
            {
                if (completed[i])
                    continue;

                else
                {
                    // [batch_size, 1, hidden_dim] => [hidden_dim]
                    Tensor curr_node = curr_nodes.subset({_c.decoder_sa().hidden_dim()}, i * _c.decoder_sa().hidden_dim());
                    if (_c.has_embedding())
                    {
                        auto ys_tmp = _project_to_vocab({curr_node});
                        uint dict_id = ys_tmp[0].data().first_item();
                        if (dict_id == 2) // <end>
                        {
                            num_completed++;
                            completed[i] = true;
                        }

                        ys[i].push_back(curr_node);
                    }
                    else
                    {
                        if (curr_node.equals_to(_get_end_node()))
                        {
                            num_completed++;
                            completed[i] = true;
                        }

                        ys[i].push_back(curr_node);
                    }

                }
            }

            curr_node_len++;
            // and then curr_nodes will be next input nodes
        }

        // combine multiple tensors to be one
        TensorList new_ys;
        for (uint i = 0; i < ys.size(); ++i)
        {
            // padding
            for (uint j = ys[i].size(); j < _c.decoder_sa().node_len(); ++j)
            {
                ys[i].push_back(_get_padding_node());
            }

            Tensor batch = Tensor::combine(ys[i]);
            new_ys.push_back(batch);
        }

        Tensor final = Tensor::combine(new_ys);
        return {final};
    }
};