#pragma once
#include "unit_test.h"
#include "../inc/3.network/transformer.h"

class TestTransformer: public TestClass
{
public:
    REGISTER_TEST_CASES( 
    test_decoder_only_1layer_train, test_decoder_only_2layer_train, test_decoder_only_2layer_inference,
    test_encoder_decoder_2layer_train, test_encoder_decoder_2layer_inference, 
    test_encoder_decoder_2layer_train_with_embedding, test_encoder_decoder_2layer_inference_with_embedding, 
    test_encoder_decoder_2layer_inference_with_position_embedding,
    test_encoder_decoder_2layer_train_with_padding);

    static void test_decoder_only_1layer_train()
    {        
        Environment::Set_Train(true); // after set_train, all the operators need to re-generate
        // test decoder_only, 1 layer, train
        Transformer::Config c(true, 1, 1, false, 1, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);

        Tensor x({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Tensor z({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});

        auto ys = trans.forward({x, z});
        /* step-1: self-attention of x
        x: [2, 3, 2]: [0,1, 2,3, 4,5, ...]
        wq/wk/wv: [1, 2, 2]: [1, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        k: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        v: [2, 1, 2, 3]: [1,5,9, 1,5,9, ..]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => [0,0,1, 0,0,1, 0,0,1, ...]
        y: [2, 1, 3, 2]: [9,9, 9,9, 9,9, ...]
        y: [2, 3, 2]: [18, 18, 18, 18, 18, 18, ...]
        y: [2, 3, 2]: [18, 19, 20, 21, 22, 23, ...]
        end of self_attention
        y: [2, 3, 2]: [37, 37, 41, 41, 45, 45, ...]
        y: [2, 3, 2]: [74, 74, 82, 82, 90, 90, ...]
        y: [2, 3, 2]: [92, 93, 102, 103, 112, 113...]
        end of fc1/fc2, and thus end of encoding
        */

        /* step-2: self-attention of z, needs mask_triangle
        x: [2, 3, 2]: [0,1, 2,3, 4,5, ...]
        wq/wk/wv: [1, 2, 2]: [1, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        k: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        v: [2, 1, 2, 3]: [1,5,9, 1,5,9, ..]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => 
        // triangle-mask: [2, -inf, -inf, 10, 50, -inf, 18,90,162, ...]
        // weights.softmax(1) => [1,0,0, 4e-18, 1, 0,  1.80485139e-35, 4.24835426e-18, 1.00000000e+00, ...]
        y: [2, 1, 3, 2]: [1,1,5,5,9,9, ...]
        y: [2, 3, 2]: [2,2,10,10, 18, 18, ...]
        y: [2, 3, 2]: [2, 3, 12, 13, 22, 23, ...]
        end of self_attention of z
        */

       /*
       step-3: cross-attention:
        x: [2, 3, 2]: [2, 3, 12, 13, 22, 23, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]

        k: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        v: [2, 1, 2, 3]: [1,5,9, 1,5,9, ..]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => [0,0,1, 0,0,1, 0,0,1, ...]
        y: [2, 1, 3, 2]: [9,9, 9,9, 9,9, ...]
        y: [2, 3, 2]: [18, 18, 18, 18, 18, 18, ...]
        y: [2, 3, 2]: [20, 21, 30, 31, 40, 41, ...]
        end of cross_attention
        y: [2, 3, 2]: [41, 41, 61, 61, 81, 81, ...]
        y: [2, 3, 2]: [82, 82, 122, 122, 162, 162, ...]
        y: [2, 3, 2]: [101, 102, 152, 153, 202, 203...]
        end of fc1/fc2, and thus end of encoding
       */
      // assert(ys.size() == 2);
      assert(ys[0].equals_to({2, 3, 2}, {102, 103, 152, 153, 202, 203, 102, 103, 152, 153, 202, 203}, 0.1));
      Environment::Set_Train(false);


/*
        // test decoder_only 2 layer, train
        // test encoder_decoder 2 layer, train

        Environment::Set_Train(false);
        Transformer trans_infer(c);
        // test decoder_only 2 layer, inference 
        ys = trans.forward({x});
        assert(ys.size() == 2);
        // test encoder_decoder 2 layer, inference 
        */
    }

    static void test_decoder_only_2layer_train()
    {        
        Environment::Set_Train(true); // after set_train, all the operators need to re-generate
        // test decoder_only, 2 layer, train
        Transformer::Config c(true, 2, 2, false, 1, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);

        Tensor x({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Tensor z({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});

        auto ys = trans.forward({x, z});
        // assert(ys.size() == 2);
        assert(ys[0].equals_to({2, 3, 2}, {2*125+450*130+62, 2*125+450*130+63, 
                                        12*125+450*130+62, 12*125+450*130+63, 
                                        22*125+450*130+62, 22*125+450*130+63,
                                        2*125+450*130+62, 2*125+450*130+63, 
                                        12*125+450*130+62, 12*125+450*130+63, 
                                        22*125+450*130+62, 22*125+450*130+63}));
        
        Environment::Set_Train(false);
        /* step-1: self-attention of x
        x: [2, 3, 2]: [0,1, 2,3, 4,5, ...]
        wq/wk/wv: [1, 2, 2]: [1, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        k: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        v: [2, 1, 2, 3]: [1,5,9, 1,5,9, ..]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => [0,0,1, 0,0,1, 0,0,1, ...]
        y: [2, 1, 3, 2]: [9,9, 9,9, 9,9, ...]
        y: [2, 3, 2]: [18, 18, 18, 18, 18, 18, ...]
        y: [2, 3, 2]: [18, 19, 20, 21, 22, 23, ...]
        end of self_attention
        y: [2, 3, 2]: [37, 37, 41, 41, 45, 45, ...]
        y: [2, 3, 2]: [74, 74, 82, 82, 90, 90, ...]
        y: [2, 3, 2]: [92, 93, 102, 103, 112, 113...]
        end of fc1/fc2, and thus end of encoding

        layer_2:

        x: [2, 3, 2]: [92, 93, 102, 103, 112, 113...]
        q/k: [2, 1, 3, 2]: [185, 185, 205, 205, 225, 225, ...]
        v: [2, 1, 2, 3]: [185,205,225, ...]
        weights: [2, 1, 3, 3]: [0,0,1, ...]
        y: [2, 1, 3, 2]: [225, ...] => [450, ...], => [92+450, 93+, 102+, 103+, 112+, 113+, ...]
        end of self_attention
        y: [185+900, .., 205+900, .., 225+900, .., ...]
        y: [185*2+900*2, .., 205*2+900*2, .., 225*2+900*2, ...]
        y: [92*5+450*5+2, +1, 102*5+450*5+2, +1, 112*5+450*5+2, +1, ...]
        end of fc1/fc2, and thus end of layer_2
        */

        /* layer_1: self-attention of z, needs mask_triangle
        x: [2, 3, 2]: [0,1, 2,3, 4,5, ...]
        wq/wk/wv: [1, 2, 2]: [1, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        k: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]
        v: [2, 1, 2, 3]: [1,5,9, 1,5,9, ..]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => 
        // triangle-mask: [2, -inf, -inf, 10, 50, -inf, 18,90,162, ...]
        // weights.softmax(1) => [1,0,0, 4e-18, 1, 0,  1.80485139e-35, 4.24835426e-18, 1.00000000e+00, ...]
        y: [2, 1, 3, 2]: [1,1,5,5,9,9, ...]
        y: [2, 3, 2]: [2,2,10,10, 18, 18, ...]
        y: [2, 3, 2]: [2, 3, 12, 13, 22, 23, ...]
        end of self_attention of z
        */

       /*
       step-3: layer_1: cross-attention:
        x: [2, 3, 2]: [2, 3, 12, 13, 22, 23, ...]
        q: [2, 1, 3, 2]: [1,1, 5,5, 9,9, ..]

        k: [2, 1, 3, 2]: [185, 185, 205, 205, 225, 225, ...]
        v: [2, 1, 2, 3]: [185,205,225, ...]
        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => [0,0,1, 0,0,1, 0,0,1, ...]
        y: [2, 1, 3, 2]: [225, ...]
        y: [2, 3, 2]: [450, ...]
        y: [2, 3, 2]: [2+450, 3+, 12+, 13+, 22+, 23+, ...]
        end of cross_attention
        y: [2, 3, 2]: [2*2+450*2+1, .., 12*2+, .., 22*2+, .., ...]
        y: [2, 3, 2]: [2*4+450*4+2, .., 12*4+, .., 22*4+, ...]
        y: [2, 3, 2]: [2*5+450*5+2, +1, 12*5+, +1, 22*5+,+1, ....]
        end of fc1/fc2, and thus end of encoding
       */

        /* layer_2: self-attention of z, needs mask_triangle
        y: [2, 3, 2]: [2*5+450*5+2, +1, 12*5+, +1, 22*5+,+1, ....]
        wq/wk/wv: [1, 2, 2]: [1, ...]
        q: [2, 1, 3, 2]: [2*10+450*10+5, .., 12*, .., 22*, .., ...]
        k: [2, 1, 3, 2]: [2*10+450*10+5, .., 12*, .., 22*, .., ...]
        v: [2, 1, 2, 3]: [2*, 12*, 22*, ..]
        weights:[1,0,0, 0,1,0, 0,0,1, ...]
        y: [2, 1, 3, 2]: [2*, 2*,12*, 12*, 22*, 22*, ...]
        y: [2, 3, 2]: [2*20+450*20+10, .., 12*, .., 22*.., ...]
        y: [2, 3, 2]: [2*25+450*25+12, +1, 12*, +1, 22*, +1, ...]
        end of self_attention of z
        */

       /*
       step-3: layer_2: cross-attention:
        x: [2, 3, 2]: [2*25+450*25+12, +1, 12*, +1, 22*, +1, ...]
        q: [2, 1, 3, 2]: [2*10+450*10+5, .., 12*, .., 22*, .., ...]

        k: [2, 1, 3, 2]: [185, 185, 205, 205, 225, 225, ...]
        v: [2, 1, 2, 3]: [185,205,225, ...]

        weights: [2, 1, 3, 3]: [2, 10, 18,  10, 50, 90, 18, 90, 162, ...] => [0,0,1, 0,0,1, 0,0,1, ...]
        y: [2, 1, 3, 2]: [225, ...]
        y: [2, 3, 2]: [450, ...]
        y: [2, 3, 2]: [2*25+450*26+12, +1, 12*, +1, 22*, +1 ...]
        end of cross_attention
        y: [2, 3, 2]: [2*50+450*52+25, .., 12*, .., 22*, .., ...]
        y: [2, 3, 2]: [2*100+450*104+50, .., 12*+, .., 22*, ...]
        y: [2, 3, 2]: [2*125+450*130+62, +1, 12*, +1, 22*, +1, .....]
        end of fc1/fc2, and thus end of encoding
       */
    }

    static void test_decoder_only_2layer_inference()
    {
        Transformer::Config c(true, 2, 2, false, 1, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::Half;

        Transformer trans(c);

        Tensor x({2, 3, 2}, {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5});

        auto ys = trans.forward({x});
        // note: didn't calculate the exact value, just check the running process
        assert(ys[0].equals_to({2, 3, 2}, {19.5, 20.5, 334, 335, 5366.59, 5367.59, 19.5, 20.5, 334, 335, 5366.59, 5367.59}, 0.1));
    }

    static void test_encoder_decoder_2layer_train()
    {        
        Environment::Set_Train(true); // after set_train, all the operators need to re-generate
        // test decoder_only, 2 layer, train
        Transformer::Config c(false, 2, 2, false, 1, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);

        Tensor x({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});
        Tensor z({2, 3, 2}, {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5});

        auto ys = trans.forward({x, z});
        // assert(ys.size() == 2);
        assert(ys[0].equals_to({2, 3, 2}, {2*125+450*130+62, 2*125+450*130+63, 
                                        12*125+450*130+62, 12*125+450*130+63, 
                                        22*125+450*130+62, 22*125+450*130+63,
                                        2*125+450*130+62, 2*125+450*130+63, 
                                        12*125+450*130+62, 12*125+450*130+63, 
                                        22*125+450*130+62, 22*125+450*130+63}));
        
        Environment::Set_Train(false);
    }

    static void test_encoder_decoder_2layer_inference()
    {
        Transformer::Config c(false, 2, 2, false, 1, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::Half;

        Transformer trans(c);

        Tensor x({2, 3, 2}, {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5});

        auto ys = trans.forward({x});
        // note: didn't calculate the exact value, just check the running process
        assert(ys[0].equals_to({2, 3, 2}, {19.5, 20.5, 334, 335, 5366.59, 5367.59, 19.5, 20.5, 334, 335, 5366.59, 5367.59}, 0.1));
    }

    static void test_encoder_decoder_2layer_train_with_embedding()
    {
        Environment::Set_Train(true); // after set_train, all the operators need to re-generate
        // test decoder_only, 2 layer, train
        Transformer::Config c(false, 2, 2, true, 3, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);
        trans._embedding.reset({3, 2}, {0, 1, 2, 3, 4, 5});

        Tensor x({2, 3}, {0, 1, 2, 0, 1, 2});
        Tensor z({2, 3}, {0, 1, 2, 0, 1, 2});

        auto ys = trans.forward({x, z});
        // assert(ys.size() == 2);
        // assert(ys[0].equals_to({2, 3}, {2, 2, 2, 2, 2, 2}));
        // assert(ys[0].equals_to({2}, {3.0667, 3.0667}, 0.1));
        
        Environment::Set_Train(false);
    }   

    static void test_encoder_decoder_2layer_inference_with_embedding()
    {
        // test decoder_only, 2 layer, train
        Transformer::Config c(false, 2, 2, true, 3, false, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);
        trans._embedding.reset({3, 2}, {0, 1, 2, 3, 4, 5});

        Tensor x({2, 3}, {0, 1, 2, 0, 1, 2});
        Tensor z({2, 3}, {0, 1, 2, 0, 1, 2});

        auto ys = trans.forward({x, z});
        assert(ys.size() == 1);
        assert(ys[0].equals_to({2, 3}, {2, 2, 2, 2, 2, 2}));
    }

    static void test_encoder_decoder_2layer_inference_with_position_embedding()
    {
        // test decoder_only, 2 layer, train
        Transformer::Config c(false, 2, 2, true, 3, true, false, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);
        trans._embedding.reset({3, 2}, {0, 1, 2, 3, 4, 5});

        Tensor x({2, 3}, {0, 1, 2, 0, 1, 2});
        Tensor z({2, 3}, {0, 1, 2, 0, 1, 2});

        auto ys = trans.forward({x, z});
        assert(ys.size() == 1);
        assert(ys[0].equals_to({2, 3}, {2, 2, 2, 2, 2, 2}));
    }

    static void test_encoder_decoder_2layer_train_with_padding()
    {
        Environment::Set_Train(true); // after set_train, all the operators need to re-generate
        // test decoder_only, 2 layer, train
        Transformer::Config c(false, 2, 2, true, 3, false, true, 2, 3, 1);
        c.decoder_sa().fc_intermediate_factor() = 1;
        c.decoder_sa().init_type() = (uint)TensorInit_Types::One;

        Transformer trans(c);
        trans._embedding.reset({3, 2}, {0, 1, 2, 3, 4, 5});

        Tensor x({2, 3}, {1, 2, 0, 1, 2, 0});
        Tensor z({2, 3}, {1, 2, 0, 1, 2, 0});

        auto ys = trans.forward({x, z});
        // assert(ys.size() == 2);
        // assert(ys[0].equals_to({2, 3}, {2, 2, 2, 2, 2, 2}));
        // assert(ys[0].equals_to({2}, {3.0667, 3.0667}, 0.1));
        // TODO: verify each step output is correct
        Environment::Set_Train(false);
    }
};