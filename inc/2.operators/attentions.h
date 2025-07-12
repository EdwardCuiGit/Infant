#pragma once

#include "operator_base.h"
#include "norm.h"
#include "fc.h"

// https://mp.weixin.qq.com/s?__biz=MjM5ODIwNjEzNQ==&mid=2649887383&idx=3&sn=bbbab92f5927fddb214684216146c7c5&chksm=bf416248c20eafb483bd7071a440acc68d6433f6e0851f906f4de702509019ddc4680cb42821&scene=27
// TODO: P0: position embedding 
// P2: labeling smoothing, drop-out, shared parameters of input/output embedding

// this supports both self attention and cross attention, for both batch mode in training and one-by-one in inference
// variant projection op: tensor<N, L, H> -> tensor<N, L, H>
class Attention : public Operator
{
    friend class TestAttention;
    friend class TestOperatorBase;
public:
    struct Config : ConfigBase
    {
        DEFINE_FIELD(uint, hidden_dim, 1);
        DEFINE_FIELD(uint, node_len, 1);
        DEFINE_FIELD(uint, multi_head, 1);
        DEFINE_FIELD(bool, is_cross_attention, false);
        // DEFINE_FIELD(bool, triangle_mask, false);
        // DEFINE_FIELD(bool, single_node, false);
        DEFINE_FIELD(float, dk, 1.0);
        DEFINE_FIELD(bool, has_fc, false);
        DEFINE_FIELD(float, fc_intermediate_factor, 4.0f);
        DEFINE_FIELD(bool, has_moe, false);
        DEFINE_FIELD(uint, total_experts, 2);
        DEFINE_FIELD(uint, active_experts, 1);
        DEFINE_FIELD(bool, has_mla, false);
        DEFINE_FIELD(uint, compression_factor, 4);
        DEFINE_FIELD(bool, has_bias, false);
        DEFINE_FIELD(uint, init_type, (uint)TensorInit_Types::Gaussian);
        DEFINE_FIELD(uint, bias_init_type, (uint)TensorInit_Types::Zero);

        // LayerNorm lm;
        DEFINE_SUB_CONFIG(LayerNorm, lm);

        Config(uint hidden_dim = 1, uint node_len = 1, uint multi_head = 1, bool is_cross_attention = false, 
        float dk = 1, bool has_fc = false, float fc_intermediate_factor = 4.0f,
        bool has_moe = false, uint total_experts = 2, uint active_experts = 1,
        bool has_mla = false,
        bool has_bias = false, 
        TensorInit_Types init_type = TensorInit_Types::Gaussian,
        TensorInit_Types bias_init_type = TensorInit_Types::Zero, const LayerNorm::Config& lm = LayerNorm::Config())
        : ConfigBase("Attention")
        {
            this->hidden_dim() = hidden_dim;
            this->node_len() = node_len;
            this->multi_head() = multi_head;
            this->is_cross_attention() = is_cross_attention;
            // this->triangle_mask() = triangle_mask;
            // this->single_node() = single_node;
            this->dk() = dk;
            this->has_fc() = has_fc;
            this->fc_intermediate_factor() = fc_intermediate_factor;
            this->has_moe() = has_moe;
            this->total_experts() = total_experts;
            this->active_experts() = active_experts;
            this->has_mla() = has_mla;
            this->has_bias() = has_bias;
            this->init_type() = (uint)init_type;
            this->bias_init_type() = (uint)bias_init_type;
            this->lm() = lm;
        }
    };

private:
    Config _c;

    // parameters
    Tensor _wq, _wk, _wv, _wp, _bq, _bk, _bv, _bp;

    // used for MLA
    Tensor _w_dq, _w_uq, _w_qr, _w_dkv, _w_uk, _w_uv, _w_kr;

    // non param, const tensor
    Tensor _right_higher_neg_inf; // note: we can put this to global const variables

    Ptr<LayerNorm> _lm, _lm1, _lm2;
    Ptr<Fc> _fc1, _fc2;
    Array<Ptr<Fc>> _fc1_moe, _fc2_moe;
    Ptr<Fc> _moe_gate;

public:
    Attention(const Config &c) : Operator(&_c, "Attention"), _c(c)
    {
        assert(c.hidden_dim() > 0);
        assert(c.node_len() > 0);
        assert(c.multi_head() > 0);
        assert(c.hidden_dim() % c.multi_head() == 0);

        if (!c.is_cross_attention()) // only self attention needs to create below
        {
            if (!_c.has_mla())
            {
                _wq = add_param("wq", {c.multi_head(), c.hidden_dim() / c.multi_head(), c.hidden_dim()}, (TensorInit_Types)c.init_type());
                _wk = add_param("wk", {c.multi_head(), c.hidden_dim() / c.multi_head(), c.hidden_dim()}, (TensorInit_Types)c.init_type());
                _wv = add_param("wv", {c.multi_head(), c.hidden_dim() / c.multi_head(), c.hidden_dim()}, (TensorInit_Types)c.init_type());

                if (c.has_bias())
                {
                    _bq = add_param("bq", {c.hidden_dim()}, (TensorInit_Types)_c.bias_init_type());
                    _bk = add_param("bk", {c.hidden_dim()}, (TensorInit_Types)_c.bias_init_type());
                    _bv = add_param("bv", {c.hidden_dim()}, (TensorInit_Types)_c.bias_init_type());
                }
            }
            else // mla
            {
                _w_dq = add_param("wdq", {c.compression_factor(), c.hidden_dim() / c.multi_head(), c.hidden_dim()}, (TensorInit_Types)c.init_type());
                _w_uq = add_param("wuq", {c.multi_head(), c.hidden_dim() / c.multi_head(), 
                    c.compression_factor(), c.hidden_dim() / c.multi_head()}, (TensorInit_Types)c.init_type());

                _w_dkv = add_param("wdkv", {c.compression_factor(), c.hidden_dim() / c.multi_head(), c.hidden_dim()}, (TensorInit_Types)c.init_type());
                _w_uk = add_param("wuk", {c.multi_head(), c.hidden_dim() / c.multi_head(), 
                    c.compression_factor(), c.hidden_dim() / c.multi_head()}, (TensorInit_Types)c.init_type());
                _w_uv = add_param("wuv", {c.multi_head(), c.hidden_dim() / c.multi_head(), 
                    c.compression_factor(), c.hidden_dim() / c.multi_head()}, (TensorInit_Types)c.init_type());
            }
        }

        _wp = add_param("wp", {c.hidden_dim(), c.hidden_dim()}, (TensorInit_Types)c.init_type());
        if (c.has_bias())
        {
            _bp = add_param("bp", {c.hidden_dim()}, (TensorInit_Types)_c.bias_init_type());
        }

        _c.lm().last_dims() = {c.hidden_dim()}; // ensure lm is for the hidden-dim
        _lm = add_op<LayerNorm>("lm", _c.lm());

        if (_c.has_fc()) // this is to init the 2 fc layers in the end
        {
            if (!_c.has_moe())
            {
                Fc::Config fc1_c(_c.hidden_dim(), (uint)(_c.hidden_dim() * _c.fc_intermediate_factor()), _c.has_bias(), (TensorInit_Types)_c.init_type(), (TensorInit_Types)_c.bias_init_type());
                Fc::Config fc2_c((uint)(_c.hidden_dim() * _c.fc_intermediate_factor()), _c.hidden_dim(), _c.has_bias(), (TensorInit_Types)_c.init_type(), (TensorInit_Types)_c.bias_init_type());
                _fc1 = add_op<Fc>("fc1", fc1_c);
                _fc2 = add_op<Fc>("fc2", fc2_c);
            }
            else // has moe
            {
                assert(_c.total_experts() > 0 && _c.active_experts() > 0 && _c.active_experts() <= _c.total_experts());
                Fc::Config fc1_c(_c.hidden_dim(), (uint)(_c.hidden_dim() * _c.fc_intermediate_factor()), _c.has_bias(), (TensorInit_Types)_c.init_type(), (TensorInit_Types)_c.bias_init_type());
                Fc::Config fc2_c((uint)(_c.hidden_dim() * _c.fc_intermediate_factor()), _c.hidden_dim(), _c.has_bias(), (TensorInit_Types)_c.init_type(), (TensorInit_Types)_c.bias_init_type());

                _fc1_moe.reserve(_c.total_experts());
                _fc2_moe.reserve(_c.total_experts());
                for (uint i = 0; i < _c.total_experts(); ++i)
                {
                    _fc1_moe[i] = add_op<Fc>(StringUtil::concat("fc1", i), fc1_c);
                    _fc2_moe[i] = add_op<Fc>(StringUtil::concat("fc2", i), fc2_c);
                }

                Fc::Config moe_gate_c(_c.hidden_dim(), _c.total_experts(), _c.has_bias(), (TensorInit_Types)_c.init_type(), (TensorInit_Types)_c.bias_init_type());
                _moe_gate = add_op<Fc>("moe_gate", moe_gate_c);
            }

            // first fc's output_dim: hidden_dim * fc_intermediate_factor
            _c.lm().last_dims() = {(uint)(_c.hidden_dim() * _c.fc_intermediate_factor())};
            _lm1 = add_op<LayerNorm>("lm1", _c.lm());

            // second fc's output_dim: hidden_dim
            _c.lm().last_dims() = {_c.hidden_dim()};
            _lm2 = add_op<LayerNorm>("lm2", _c.lm());
        }

        // if (_c.triangle_mask())
        {
            _right_higher_neg_inf = add_param("right_higher_neg_inf", {c.node_len(), c.node_len()}, TensorInit_Types::RIGHT_HIGHER_NEG_INF);
        }
    }

    /* self_attention 
    input for multi_node:
        xs[0]: x: [batch_size, node_len, hidden_dim], is the whole node sequence with fixed length after padding
    input for single_node: 
        xs[0]: x: [batch_size, 1, hidden_dim]
        ys[2]: k_cache: [batch_size, multi_head, curr_node_len, hidden_dim], start from empty
        ys[3]: v_cache: [batch_size, multi_head, hidden_dim, curr_node_len], start from empty
    output: ys
        y: [batch_size, node_len_input or 1, hidden_dim]
        q: [batch_size, multi_head, node_len_input or 1, hidden_dim/multi_head]
        k: [batch_size, multi_head, node_len_input or node_len_cache or node_len_encoder, hidden_dim/multi_head]
        v: [batch_size, multi_head, hidden_dim/multi_head, node_len_input or node_len_cache or node_len_encoder]
    cross_attention:
    input: output above self_attention
    output:
        ys[0]: y: [batch_size, node_len, hidden_dim]
    */
    virtual TensorList forward(const TensorList &xs, const RTConfig& rc) const override
    {
        bool triangle_mask = rc.access_bool("triangle_mask");
        bool has_fc = rc.access_bool("has_fc");
        bool single_node = rc.access_bool("single_node");
        return forward(xs, triangle_mask, has_fc, single_node);
    }

    virtual TensorList forward(const TensorList &xs, bool triangle_mask = false, bool has_fc = true, bool single_node = false) const
    {
        /*// trucation or padding, no need to do here as it can be done during data loading
        assert(xs.size() > 0);
        Tensor x = xs[0];
        assert(x.shape() == 3); // [batch_size, node_len, hidden_dim]
        if (!_c.single_node()) // no need to do padding or truncation for single node
        {
            uint node_len = x.dim()[1];
            if (node_len > _c.node_len()) // truncation
            {
                
            }
            else if (node_len < _c.node_len()) // padding
            {

            }
        }*/

        TensorList ys;

        if (!_c.is_cross_attention())
        {
            ys = this->_self_attention(xs, triangle_mask, single_node);
        }
        else
        {
            ys = this->_cross_attention(xs, single_node);
        }

        // _c.has_fc means this attention has parameters, has_fc means this needs to run fc, 
        if (_c.has_fc() && has_fc)
        {
            // fc1 + relu + add + norm2
            assert(ys.size() > 0); 

            if (!_c.has_moe())
            {
                Tensor intermediate = _lm1->forward(_fc1->forward(ys[0]).activation_(Activation_Types::Relu));
                ys[0] = _lm2->forward(_fc2->forward(intermediate).activation_(Activation_Types::Relu).add_(ys[0]));
            }
            else // moe
            {
                Tensor gates = _moe_gate->forward(ys[0]).softmax(1); // [batch_size, node_len, total_experts]
                TensorList topk = gates.topk(_c.active_experts()); // [batch_size, node_len, active_experts]
                Tensor topk_ids = topk[0].merge_dim(0, 2);// [batch_size* node_len, active_experts]
                Tensor topk_weights = topk[1].merge_dim(0, 2); // [batch_size  * node_len, active_experts]
                Tensor hidden_states = ys[0].merge_dim(0, 2); // [batch_size * node_len, hidden_dim]
                Tensor output_total(hidden_states.dim(), TensorInit_Types::Zero);

                for (uint i = 0; i < _c.total_experts(); ++i)
                {
                    TensorList topk_ids_per_expert = topk_ids.where(CompareTypes::Equal, i); 
                    Tensor sample_ids = topk_ids_per_expert[0]; // [sub of batch_size * node_len], value range is b * n
                    Tensor expert_ids = topk_ids_per_expert[1]; // [sub of batch_size * node_len], value range is active_experts
                    Tensor selected_samples = hidden_states.index({sample_ids}).unsqueeze(1); //[sub batch_size*node_len, 1, hidden_dim]

                    Tensor intermediate = _lm1->forward(_fc1_moe[i]->forward(selected_samples).activation_(Activation_Types::Relu));
                    Tensor output = _lm2->forward(_fc2_moe[i]->forward(intermediate).activation_(Activation_Types::Relu).add_(selected_samples));
                    output = output.squeeze(); // [sub batch_size]*node_len, hidden_dim]
                    Tensor selected_weights = topk_weights.index({sample_ids, expert_ids}).unsqueeze(1); // [sub batch_size*node_len, 1]                    
                    Tensor weighted_output = output.mul(selected_weights, 1.0f, 0.0f, 1, 0).squeeze(2); // [sub b * n, hidden_dim]
                    
                    // output_total.non_contiguous_index(sample_ids).add_(weighted_output);
                    weighted_output = output_total.index({sample_ids}).add(weighted_output);
                    output_total = output_total.assign(weighted_output, sample_ids);
                }

                ys[0] = output_total.reshape(ys[0].dim());
            }
        }

        return ys;
    }

private:
    virtual bool is_const() const
    {
        return !_c.has_moe();
    }

    void _generate_QKV(const Tensor &x, Tensor &q, Tensor &k, Tensor &v) const
    {
        q = x.dot(_wq);
        k = x.dot(_wk);
        v = x.dot(_wv);
        if (_c.has_bias())
        {
            q.add_(_bq);
            k.add_(_bk);
            v.add_(_bv);
        }

    }

    // note: below treats multi-node and single node shape are the same
    // self_attention: support both single node and multi node, below inline comments are for self_attentions
    // cross-attention: support both single node and multinode
    /* multi node: 
        y_sa: [batch_size, multi_head, node_len_decoder, hidden_dim/multi_head]
        q_sa: [batch_size, multi_head, node_len_decoder, hidden_dim/multi_head]
        k_encoder: [batch_size, multi_head, node_len_encoder, hidden_dim/multi_head]
        weights: [batch_size, multi_head, node_len_decoder, node_len_encoder]
        v_encoder: [batch_size, multi_head, hidden_dim/multi_head, node_len_encoder]
        y: [batch_size, multi_head, node_len_decoder, hidden_dim/multi_head]
        y: [batch_size, node_len_decoder, multi_head, hidden_dim/multi_head]
        y: [batch_size, node_len_decoder, hidden_dim]
    */
    /* single node: during inference 
        q: [batch_size, multi_head, 1, hidden_dim/multi_head]
        k_encoder: [batch_size, multi_head, node_len_encoder, hidden_dim/multi_head]
        weights_ca: [batch_size, multi_head, node_len_encoder]
        v_encoder: [batch_size, multi_head, hidden_dim/multi_head, node_len_encoder],
        y_ca: [batch_size, 1, multi_head, hidden_dim/multi_head]
        y_ca: [batch_size, 1, hidden_dim]
    */
    Tensor _run_QKV(const Tensor &x, const Tensor &q, 
    const Tensor &k, const Tensor &v, const Tensor &xq_padding_mask = Tensor(), const Tensor &kv_padding_mask = Tensor(),
    bool triangle_mask = false) const
    {
        // [batch_size, multi_head, node_len, hidden_dim/multi_head] * [batch_size, multi_head, node_len, hidden_dim/multi_head] 
        // => [batch_size, multi_head, xq_node_len, kv_node_len]
        Tensor weights;
        if (!_c.has_mla())
        {
            weights = q.dot(k, 2);                 
        }
        else
        {
            // note: we shall merge _w_uq and_w_uk to be one tensor, so that we have less dots
            // in this case, q is c_Q, k is ckv
            Tensor c_Q = q, ckv = k;
            // [batch_size, node_len, c.compression_factor(), c.hidden_dim() / c.multi_head()] * {c.multi_head(), c.hidden_dim() / c.multi_head(), c.compression_factor(), c.hidden_dim() / c.multi_head()} => 
            // [batch_size, node_len, multi_head, c.hidden_dim() / c.multi_head()]
            Tensor q_C = c_Q.dot(_w_uq, 0, 2).swap(1, 2); // [batch_size, multi_head, node_len, hidden_dim/multi_head]
            // ckv: {batch_size, node_len, c.compression_factor(), c.hidden_dim() / c.multi_head()} * {c.multi_head(), c.hidden_dim() / c.multi_head(), c.compression_factor(), c.hidden_dim() / c.multi_head()} => 
            // [batch_size, node_len, multi_head, c.hidden_dim() / c.multi_head()]
            Tensor k_C = ckv.dot(_w_uk, 0, 2).swap(1, 2); // [batch_size, multi_head, node_len, hidden_dim/multi_head]
            weights = q_C.dot(k_C, 2); // [batch_size, multi_head, node_len, node_len]

        }

        // this will make the weights value to be -inf if it's padding
        if (xq_padding_mask.size() > 0 && kv_padding_mask.size() > 0) // [batch_size, node_len] : 1 or 0
        {
            /* 3rd token is masked
            1, 1, -inf
            1, 1, -inf
            -inf, -inf, -inf
            */
            Tensor mask_matrix = xq_padding_mask.dot(kv_padding_mask, 1, 0).replace(1.0f, 0.0f, INF_NEG); 
            // [batch_size, xq_node_len, kv_node_len]
            weights.add_(mask_matrix, 1, 1, 0, 1, 2);
        }

        // triangle mask: used in decoder's self attention
        /* nodes: A B C
        A, -inf, -inf
        A B, -inf
        A B C
        */
        // note: this is only used for training
        if (triangle_mask)
        {
            weights.add_(_right_higher_neg_inf, 1 / std::sqrt(_c.dk()), 1, 0, 0, 2);
        }
        else
        {
            weights.linear_(1 / std::sqrt(_c.dk()));
        }

        // padding item's weights will be 0 after softmax
        // note: there could be drop out
        weights.softmax_(1); // no shape change

        // [batch_size, multi_head, node_len, node_len] * [batch_size, multi_head, hidden_dim/multi_head, node_len]
        // => [batch_size, multi_head, node_len, hidden_dim/mlti_head]
        Tensor y = weights.dot(v, 2); 
        y = y.swap(1, 2).merge_dim(2, 2); // [batch_size, node_len, hidden_dim]

        // projection
        y = y.dot(_wp);
        if (_c.has_bias())
            y = y.add(_bp); // padding items will bave bias

        // residual
        y.add_(x); // padding items will have residual
        
        // lm
        y = _lm->forward(y); // padding items will be normalized

        return y; // [batch_size, *node_len, hidden_dim]
    }

    /* 
    1. multi_node:
    input: 
        xs[0]: x: [batch_size, node_len, hidden_dim], is the whole node sequence with fixed length after padding
    output: ys
        y: [batch_size, node_len, hidden_dim]
        q: [batch_size, multi_head, node_len, hidden_dim/multi_head]
        k: [batch_size, multi_head, node_len, hidden_dim/multi_head]
        v: [batch_size, multi_head, hidden_dim/multi_head, node_len]
    2. single_node:
    input: 
        xs[0]: x: [batch_size, 1, hidden_dim]
        xs[1]: k_cache: [batch_size, multi_head, curr_node_len, hidden_dim], start from empty
        xs[2]: v_cache: [batch_size, multi_head, hidden_dim, curr_node_len], start from empty
    output: ys
        y: [batch_size, 1, hidden_dim]
        q: [batch_size, multi_head, 1, hidden_dim/multi_head]
        k_cache: [batch_size, multi_head, curr_node_len, hidden_dim/multi_head]
        v_cache: [batch_size, multi_head, hidden_dim/multi_head, curr_node_len]
    */
    TensorList _self_attention(const TensorList &xs, bool triangle_mask, bool single_node = false) const
    {
        if (!single_node) // this is common self attention: encoder's, decoder's in training
        {
            assert(xs.size() >= 1);
            Tensor x = xs[0];
            assert(x.shape() == 3);
            assert(x.dim()[1] == _c.node_len());
            assert(x.dim()[2] == _c.hidden_dim());
            Tensor x_padding_mask;
            if (xs.size() >= 2)
            {
                x_padding_mask = xs[1];
            }

            // x:[batch_size, node_len, hidden_dim].dot(wq/wk/wv:[multi_head, hidden_dim/multi_head, hidden_dim])
            // -> q/k/v: [batch_size, node_len, multi_head, hidden_dim/multi_head]
            Tensor q, k, v;
            this->_generate_QKV(x, q, k, v); 

            q = q.swap(1, 2);            // [batch_size, multi_head, node_len, hidden_dim/multi_head]
            k = k.swap(1, 2);            // [batch_size, multi_head, node_len, hidden_dim/multi_head]
            v = v.move_forward(2, 2, 1); // [batch_size, multi_head, hidden_dim/multi_head, node_len]

            Tensor y = this->_run_QKV(x, q, k, v, x_padding_mask, x_padding_mask, triangle_mask);

            // during encoder decoder cross attention, encoder's q, k, v are used
            return {y, q, k, v};
        }
        else // single-node, decoder's in inference, node by node, needs kv-cache
        {
            assert(xs.size() >= 1);
            Tensor x = xs[0]; // [batch_size, 1, hidden_dim]
            assert(x.shape() == 3);
            uint batch_size = x.dim()[0];
            assert(x.dim()[1] == 1);
            assert(x.dim()[2] == _c.hidden_dim());

            Tensor q, k, v, y;
            if (!_c.has_mla())
            {
                this->_generate_QKV(x, q, k, v);// [batch_size, 1, multi_head, hidden_dim/multi_head]

                assert(xs.size() >= 3);
                // these are used as both inputs and outputs
                Tensor k_cache = xs[1]; // [batch_size, multi_head, curr_node_len, hidden_dim/multi_head]
                Tensor v_cache = xs[2]; // [batch_size, multi_head, hidden_dim/multi_head, curr_node_len]
                if (k_cache.size() > 0)
                {
                    assert(k_cache.dim()[0] == batch_size && v_cache.dim()[0] == batch_size);
                    assert(k_cache.dim()[1] == _c.multi_head() && v_cache.dim()[1] == _c.multi_head());
                    assert(k_cache.dim()[3] == _c.hidden_dim() && v_cache.dim()[2] == _c.hidden_dim()); 
                    assert(k_cache.dim()[2] == v_cache.dim()[3]); // curr_node_len

                    // needs to append new k_cache, v_cache first, in the start, k/v cache is empty tensor
                    k_cache.move_forward_(2, 1, 0); // [curr_node_len, batch_size, multi_head, hidden_dim/multi_head]
                    v_cache.move_forward_(3, 1, 0); // [curr_node_len, batch_size, multi_head, hidden_dim/multi_head]
                }

                k = k.merge_dim(0, 2); // [batch_size, multi_head, hidden_dim/multi_head]
                k_cache.append_(k);
                k_cache.move_forward_(1, 2, 0); // [batch_size, multi_head, curr_node_len, hidden_dim/multi_head]

                v = v.merge_dim(0, 2); // [batch_size, multi_head, hidden_dim/multi_head]
                v_cache.append_(v);
                v_cache.move_forward_(1, 3, 0); // [batch_size, multi_head, hidden_dim/multi_head, curr_node_len]
                k = k_cache;
                v = v_cache;
                // note: single_mode no need to do padding mask, as this is inference mode's decoder, there is no padding tokens
                y = this->_run_QKV(x, q, k, v, Tensor(), Tensor(), triangle_mask);
            }
            else
            {
                Tensor cq = x.dot(_w_dq); // [batch_size, 1, hidden_dim] dot [c.compression_factor(), c.hidden_dim() / c.multi_head(), c.hidden_dim()] => [batch_size, 1, c.compression_factor(), c.hidden_dim() / c.multi_head()]
                Tensor ckv = x.dot(_w_dkv); // {batch_size, node_len, hidden_dim} dot {c.compression_factor(), c.hidden_dim() / c.multi_head(), c.hidden_dim()} => {batch_size, node_len, c.compression_factor(), c.hidden_dim() / c.multi_head()}

                assert(xs.size() >= 2);
                Tensor ckv_cache = xs[1]; // [batch_size, node_len, c.compression_factor(), c.hidden_dim() / c.multi_head]
                ckv_cache = ckv_cache.merge(ckv, 1); // [batch_size, node_len + 1, c.compression_factor(), c.hidden_dim() / c.multi_head]
                v = ckv_cache.dot(_w_uv); // [batch_size, node_len, c.compression_factor(), c.hidden_dim() / c.multi_head] dot [c.multi_head(), c.hidden_dim() / c.multi_head(), c.compression_factor(), c.hidden_dim() / c.multi_head] => 
                // [batch_size, node_len, multi_head, c.hidden_dim() / c.multi_head]
                v = v.move_forward(2, 2, 1); // [batch_size, multi_head, c.hidden_dim() / c.multi_head, node_len]
                y = this->_run_QKV(x, cq, ckv_cache, v, Tensor(), Tensor(), triangle_mask);
            }

            return {y, q, k, v};
        }
    }

    /*
    input: 
        xs[0]: x: [batch_size, node_len_decoder, hidden_dim]
        xs[1]: q: [batch_size, multi_head, node_len_decoder, hidden_dim/multi_head]
        xs[2]: k: [batch_size, multi_head, node_len_encoder, hidden_dim/multi_head]
        xs[3]: v: [batch_size, multi_head, hidden_dim/multi_head, node_len_encoder]
    output:
        ys[0]: [batch_size, node_len, hidden_dim]
    */ 

    // note: multi-node and single-node process are totally the same
    // note: we assume node_len for encoder and decoder are the same, need padding, but the node_len_decoder == 1
    TensorList _cross_attention(const TensorList &xs, bool single_node = false) const
    {
        // note: cross attention used encoder's last layer, instead of each layer
        // note: cross attention does not use decoder tokens for attention
        // note: intermediate size is not considered
        // TODO: assert on x/q/k/v dims
        assert(xs.size() >= 4);
        Tensor x = xs[0];
        Tensor q = xs[1];
        Tensor k = xs[2];
        Tensor v = xs[3];

        Tensor xq_padding_mask, kv_padding_mask;
        if (xs.size() >= 6)
        {
            xq_padding_mask = xs[4];
            kv_padding_mask = xs[5];
        }

        assert(x.shape() == 3 && q.shape() == 4 && k.shape() == 4 && v.shape() == 4);
        // check batch_size
        assert(x.dim()[0] == q.dim()[0] && q.dim()[0] == k.dim()[0] && k.dim()[0] == v.dim()[0]); // batch_size
        // check node_len
        if (!single_node)
            assert(x.dim()[1] == q.dim()[2] && q.dim()[2] == k.dim()[2] && k.dim()[2] == v.dim()[3] && v.dim()[3] == _c.node_len()); // node_len
        else
            assert(x.dim()[1] == 1 && q.dim()[2] == 1 && k.dim()[2] == v.dim()[3] && v.dim()[3] == _c.node_len()); // node_len
        // check hidden_dim
        assert(x.dim()[2] == q.dim()[3] * _c.multi_head() && x.dim()[2] == k.dim()[3] *_c.multi_head()
        && x.dim()[2] == v.dim()[2] * _c.multi_head() && x.dim()[2] == _c.hidden_dim());
        // check multi_head
        assert(_c.multi_head() == q.dim()[1] && _c.multi_head() == k.dim()[1] && _c.multi_head() == v.dim()[1]);

        Tensor y = this->_run_QKV(x, q, k, v, xq_padding_mask, kv_padding_mask, false);
        return {y};
    }
};