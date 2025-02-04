#pragma once
#include "unit_test.h"
#include "../inc/2.operators/attentions.h"

class TestAttention: public TestClass
{
public:
    REGISTER_TEST_CASES(test_self_attention, test_self_attention_multi_head, test_self_attention_triangle_mode, 
    test_self_attention_single_node, test_cross_attention, test_cross_attention_single_node, test_has_bias, test_has_lm, 
    test_dk, test_transformer_fc)

    static void test_self_attention()
    {
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Attention op(Attention::Config(2, 3, 1, false, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));

        TensorList ys;
        Tensor q, k, v;
        op._generate_QKV(x, q, k, v);
        assert(q.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));
        assert(k.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));
        assert(v.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));

        q = q.swap(1, 2);            // [1,1, 5,5, 9,9,  ...]
        k = k.swap(1, 2);            // [1,1, 5,5, 9,9,  ...]
        v = v.move_forward(2, 2, 1); // [2, 1, 2, 3] => [1,5,9, 1,5,9, ...]
        Tensor y = op._run_QKV(x, q, k, v);
        // weights = q.dot(k, 2) => [2, 1, 3, 3] => [2,10,18, 10,50,90, 18,90,162,  ...]
        // weights.softmax(1) => [1.12497423e-07, 3.35350093e-04, 9.99664537e-01, 1.80485139e-35, 4.24835426e-18, 1.00000000e+00,
        // 2.89464031e-63, 5.38018616e-32, 1.00000000e+00, ...]
        // y = weights.dot(v, 2) => [2, 1, 3, 2] => [9, 9, 9, 9, 9, 9,  ...]
        // y.dot(wp) => [2, 3, 2] => [18, 18, 18, 18, 18, 18, ...]
        // residual => [18, 19, 20, 21, 22, 23, ...]
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.data().vector().equals_to({17.997315, 18.997315, 20, 21, 22, 23, 17.997315, 18.997315, 20, 21, 22, 23}));

        ys = op.forward({x});
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 3, 2}));
        assert(ys[0].data().vector().equals_to({17.997315, 18.997315, 20, 21, 22, 23, 17.997315, 18.997315, 20, 21, 22, 23}));
    }

    static void test_self_attention_multi_head()
    {
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Attention op(Attention::Config(2, 3, 2, false, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));

        TensorList ys;
        Tensor q, k, v;
        op._generate_QKV(x, q, k, v); 
        // wq/wk/wv: [2, 1, 2]
        // [2, 3, 2, 1] [batch_size, node_len, multi_head, hidden_dim/multi_head]        
        assert(q.data().vector().equals_to({1,1, 5,5, 9,9,  1, 1, 5, 5, 9, 9}));
        assert(k.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));
        assert(v.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));

        q = q.swap(1, 2);            // [1,5,9, 1,5,9,  ...] [2, 2, 3, 1] [batch_size, multi_head, node_len, hidden_dim/multi_head]
        k = k.swap(1, 2);            // [1,5,9, 1,5,9,  ...] [2, 2, 3, 1]
        v = v.move_forward(2, 2, 1); // [2, 2, 1, 3] => [1,5,9, 1,5,9, ...] [batch_size, multi_head, hidden_dim/multi_head, node_len]
        Tensor y = op._run_QKV(x, q, k, v);
        // weights = q.dot(k, 2) => [2, 2, 3, 3] => [1,5,9, 5,25,81, 9,45,81,  1,5,9, 5,25,81, 9,45,81, ...] 
        // weights.softmax(1) => []
        // y = weights.dot(v, 2) => [2, 2, 3, 1] => [9, 9, 9, 9, 9, 9,  ...] [batch_size, multi_head, node_len, hidden_dim/multi_head]
        // y.dot(wp) => [2, 3, 2] => [18, 18, 18, 18, 18, 18, ...]
        // residual => [18, 19, 20, 21, 22, 23, ...]
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.data().vector().equals_to({17.85089, 18.850889, 20, 21, 22, 23, 17.85089, 18.850889, 20, 21, 22, 23}, 0.01));

        ys = op.forward({x});
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 3, 2}));
        assert(ys[0].data().vector().equals_to({17.85089, 18.850889, 20, 21, 22, 23, 17.85089, 18.850889, 20, 21, 22, 23}, 0.01));
    }

    static void test_self_attention_triangle_mode()
    {
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Attention op(Attention::Config(2, 3, 1, false, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));

        TensorList ys;
        Tensor q, k, v;
        op._generate_QKV(x, q, k, v);
        assert(q.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));
        assert(k.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));
        assert(v.data().vector().equals_to({1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9}));

        q = q.swap(1, 2);            // [1,1, 5,5, 9,9,  ...]
        k = k.swap(1, 2);            // [1,1, 5,5, 9,9,  ...]
        v = v.move_forward(2, 2, 1); // [2, 1, 2, 3] => [1,5,9, 1,5,9, ...]
        Tensor y = op._run_QKV(x, q, k, v, Tensor(), Tensor(), true);
        // weights = q.dot(k, 2) => [2, 1, 3, 3] => [2,10,18, 10,50,90, 18,90,162,  ...]
        // triangle-mask: [2, -inf, -inf, 10, 50, -inf, 18,90,162, ...]
        // weights.softmax(1) => [1,0,0, 4e-18, 1, 0,  1.80485139e-35, 4.24835426e-18, 1.00000000e+00, ...]
        // y = weights.dot(v, 2) => [2, 1, 3, 2] => [1, 1, 5, 5, 9, 9,  ...]
        // y.dot(wp) => [2, 3, 2] => [2, 2, 10, 10, 18, 18, ...]
        // residual => [2, 3, 12, 13, 22, 23, ...]
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.data().vector().equals_to({2, 3, 12, 13, 22, 23,   2, 3, 12, 13, 22, 23}, 0.01));

        ys = op.forward({x}, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 3, 2}));
        assert(y.data().vector().equals_to({2, 3, 12, 13, 22, 23,   2, 3, 12, 13, 22, 23}, 0.01));
    }

    static void test_self_attention_single_node()
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});
        TensorList ys(4);

        Attention op(Attention::Config(2, 1, 1, false, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));
        // x: [2, 2], wq/wk/wv: [1, 2, 2]
        // q/k/v: [2, 1, 2] => [1, 1, 1, 1]
        // k_cache: [2, 1, 1, 2] => [1, 1, 1, 1]
        // v_cache: [2, 1, 2, 1] => [1, 1, 1, 1]
        // weights: [2, 1, 1] => [2, 2]
        // weights: [2, 1, 1] => [1, 1]
        // y: [2, 1, 2] => [1, 1, 1, 1]
        // y: [2, 2] => [2, 2, 2, 2] dot
        // y: [2, 2] => [2, 3, 2, 3] add

        ys = op.forward({x, Tensor(), Tensor()}, false, false, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({2, 3, 2, 3}));
        assert(ys[1].dim().equals_to({2, 1, 1, 2}));
        assert(ys[1].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[2].dim().equals_to({2, 1, 1, 2}));
        assert(ys[2].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[3].dim().equals_to({2, 1, 2, 1}));
        assert(ys[3].data().vector().equals_to({1, 1, 1, 1}));
    }

    /*
    input: 
        xs[0]: x: [batch_size, node_len_decoder, hidden_dim]
        xs[1]: q: [batch_size, multi_head, node_len_decoder, hidden_dim/multi_head]
        xs[2]: k: [batch_size, multi_head, node_len_encoder, hidden_dim/multi_head]
        xs[3]: v: [batch_size, multi_head, hidden_dim/multi_head, node_len_encoder]
    output:
        ys[0]: y: [batch_size, node_len, hidden_dim]
    */ 
    static void test_cross_attention()
    {
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});

        Tensor q({2, 1, 3, 2});
        q.data().vector().set(0, {1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9});
        
        Tensor k({2, 1, 3, 2});
        k.data().vector().set(0, {1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9});

        Tensor v({2, 1, 2, 3});
        v.data().vector().set(0, {1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9});

        Attention op(Attention::Config(2, 3, 1, true, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));
        TensorList ys;
        ys = op.forward({x, q, k, v});
        // weights = q.dot(k, 2): [2, 1, 3, 3] => [2, 10, 18,  10, 50, 90,  18, 90, 162, ...]
        // softmax: => [0, 0, 1,  0, 0, 1,  0, 0, 1, ...]
        // weights.dot(v, 2): [2, 1, 3, 2] => [9, 9,  9, 9,  9, 9, ...]
        // y: swap/merge: [2, 3, 2]
        // dot: [18, 18, ...]
        // add: [18, 19, 20, 21, 22, 23, ...]
        assert(ys[0].dim().equals_to({2, 3, 2}));
        assert(ys[0].data().vector().equals_to({17.997315, 18.997315, 20, 21, 22, 23, 17.997315, 18.997315, 20, 21, 22, 23}));
    }

    static void test_cross_attention_single_node()
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});

        Tensor q({2, 1, 1, 2});
        q.data().vector().set(0, {1, 1, 1, 1});
        
        Tensor k({2, 1, 3, 2});
        k.data().vector().set(0, {1, 1, 5, 5, 9, 9, 1, 1, 5, 5, 9, 9});

        Tensor v({2, 1, 2, 3});
        v.data().vector().set(0, {1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9});

        Attention op(Attention::Config(2, 3, 1, true, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));
        TensorList ys;
        ys = op.forward({x, q, k, v}, false, false, true);
        // weights = q.dot(k, 2): [2, 1, 1, 3] => [2, 10, 18, ...]
        // softmax: => [0, 0, 1, ...]
        // weights.dot(v, 2): [2, 1, 1, 2] => [9, 9, ...]
        // y: swap/merge: [2, 1, 2]
        // dot: [18, 18, ...]
        // add: [18, 19, ...]
        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({17.997315, 18.997315, 17.997315, 18.997315}));
    }

    static void test_has_bias() // self_attention_single_node
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});

        Attention op(Attention::Config(2, 1, 1, false, 1.0, false, 4, true, TensorInit_Types::One, TensorInit_Types::One));
        // x: [2, 2], wq/wk/wv: [1, 2, 2]
        // q/k/v: [2, 1, 2] => [2, 2, 2, 2]
        // k_cache: [2, 1, 1, 2] => [2, 2, 2, 2]
        // v_cache: [2, 1, 2, 1] => [2, 2, 2, 2]
        // weights: [2, 1, 1] => [8, 8]
        // weights: [2, 1, 1] => [1, 1]
        // y: [2, 1, 1, 2] => [2, 2, 2, 2]
        // y: [2, 2] => [5, 5, 5, 5] dot with bias
        // y: [2, 2] => [5, 6, 5, 6] add

        TensorList ys = op.forward({x, Tensor(), Tensor()}, false, false, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({5, 6, 5, 6}));
        assert(ys[1].dim().equals_to({2, 1, 1, 2}));
        assert(ys[1].data().vector().equals_to({2, 2, 2, 2}));
        assert(ys[2].dim().equals_to({2, 1, 1, 2}));
        assert(ys[2].data().vector().equals_to({2, 2, 2, 2}));
        assert(ys[3].dim().equals_to({2, 1, 2, 1}));
        assert(ys[3].data().vector().equals_to({2, 2, 2, 2}));
    }

    static void test_has_lm() // self attention single node
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});
        auto lm_config = LayerNorm::Config({2}, true, 0.1, true, true);

        Attention op(Attention::Config(2, 1, 1, false, 1.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero, lm_config));
        // x: [2, 2], wq/wk/wv: [1, 2, 2]
        // q/k/v: [2, 1, 2] => [1, 1, 1, 1]
        // k_cache: [2, 1, 1, 2] => [1, 1, 1, 1]
        // v_cache: [2, 1, 2, 1] => [1, 1, 1, 1]
        // weights: [2, 1, 1] => [2, 2]
        // weights: [2, 1, 1] => [1, 1]
        // y: [2, 1, 2] => [1, 1, 1, 1]
        // y: [2, 2] => [2, 2, 2, 2] dot
        // y: [2, 2] => [2, 3, 2, 3] add

        TensorList ys = op.forward({x, Tensor(), Tensor()}, false, false, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({-1, 1, -1, 1})); // only y is changed
        assert(ys[1].dim().equals_to({2, 1, 1, 2}));
        assert(ys[1].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[2].dim().equals_to({2, 1, 1, 2}));
        assert(ys[2].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[3].dim().equals_to({2, 1, 2, 1}));
        assert(ys[3].data().vector().equals_to({1, 1, 1, 1}));
    }

    static void test_dk() // self attention single node
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});

        Attention op(Attention::Config(2, 1, 1, false, 4.0, false, 4, false, TensorInit_Types::One, TensorInit_Types::Zero));
        // x: [2, 2], wq/wk/wv: [1, 2, 2]
        // q/k/v: [2, 1, 2] => [1, 1, 1, 1]
        // k_cache: [2, 1, 1, 2] => [1, 1, 1, 1]
        // v_cache: [2, 1, 2, 1] => [1, 1, 1, 1]
        // weights: [2, 1, 1] => [1, 1] => here changed
        // weights: [2, 1, 1] => [1, 1]
        // y: [2, 1, 2] => [1, 1, 1, 1]
        // y: [2, 2] => [2, 2, 2, 2] dot
        // y: [2, 2] => [2, 3, 2, 3] add

        TensorList ys = op.forward({x, Tensor(), Tensor()}, false, false, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({2, 3, 2, 3}));
        assert(ys[1].dim().equals_to({2, 1, 1, 2}));
        assert(ys[1].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[2].dim().equals_to({2, 1, 1, 2}));
        assert(ys[2].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[3].dim().equals_to({2, 1, 2, 1}));
        assert(ys[3].data().vector().equals_to({1, 1, 1, 1}));
    }

    static void test_transformer_fc() // self attention single node has_lm
    {
        Tensor x({2, 1, 2}, TensorInit_Types::Ordinal);
        x.data().vector().set(0, {0, 1, 0, 1});

        Attention op(Attention::Config(2, 1, 1, false, 1.0, true, 1, false, TensorInit_Types::One, TensorInit_Types::Zero));
        // x: [2, 2], wq/wk/wv: [1, 2, 2]
        // q/k/v: [2, 1, 2] => [1, 1, 1, 1]
        // k_cache: [2, 1, 1, 2] => [1, 1, 1, 1]
        // v_cache: [2, 1, 2, 1] => [1, 1, 1, 1]
        // weights: [2, 1, 1] => [2, 2]
        // weights: [2, 1, 1] => [1, 1]
        // y: [2, 1, 2] => [1, 1, 1, 1]
        // y: [2, 2] => [2, 2, 2, 2] dot
        // y: [2, 2] => [2, 3, 2, 3] add
        // fc1: [5, 5, 5, 5]
        // fc2: [10+2, 10+3, 10+2, 10+3]

        TensorList ys = op.forward({x, Tensor(), Tensor()}, false, true, true);
        assert(ys.size() == 4);

        assert(ys[0].dim().equals_to({2, 1, 2}));
        assert(ys[0].data().vector().equals_to({12, 13, 12, 13})); // only y is changed
        assert(ys[1].dim().equals_to({2, 1, 1, 2}));
        assert(ys[1].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[2].dim().equals_to({2, 1, 1, 2}));
        assert(ys[2].data().vector().equals_to({1, 1, 1, 1}));
        assert(ys[3].dim().equals_to({2, 1, 2, 1}));
        assert(ys[3].data().vector().equals_to({1, 1, 1, 1}));
    }
};