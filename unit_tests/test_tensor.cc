#pragma once
#include "unit_test.h"
#include "inc/0.tensors/tensor.h"

class TestTensor : public TestClass
{
public:
    REGISTER_TEST_CASES(test_create, test_add, test_mul, test_dot, test_mse, test_ce, test_euclidean, test_linear, test_sqrt, test_pow, test_softmax, test_activation, test_sum, test_avg, test_var, test_max, test_min, test_swap, 
                        test_move_forward, test_im2col, test_im2col_grad, test_merge_dim, test_divide, test_combine, test_inflate, test_squeeze, test_subset,
                        test_deep_copy, test_equals_to, test_encode_by_dict, test_search_by_dict, test_decode, test_load, test_save, test_append, test_map,
                        test_unsqueeze, test_reshape, test_where, test_topk, test_index, test_index_non_cross, test_assign, test_dropout, test_norm_ln, test_rms_norm, test_rope,
                        test_replace, test_insert);

    static void test_create()
    {
        TensorD<float> x1;
        assert(x1.size() == 0);
        assert(x1.shape() == 0);
        TensorD<float> x2({2, 3, 5}, TensorInit_Types::One);
        assert(x2.size() == 30);
        assert(x2.shape() == 3);
        assert(x2.dim()[1] == 3);
        assert(x2.vector().sum() == 30);
        TensorD<float> x3(x2);
        assert(x3._vector->sum() == 30);

        assert(x3.dim_to_size(1) == 15);
        assert(x3.dim_to_size(1, 1) == 3);
        assert(x3.dim_to_size(1, 2, false) == 6);
        assert(x3.size_to_dim(6) == 2);
        assert(x3.size_to_dim(5, false) == 1);
        x3.clear();
        assert(x3.size() == 0);

        TensorD<float> x4({3, 3}, TensorInit_Types::LEFT_LOWER_ONE);
        assert(x4.vector().equals_to({1, 0, 0, 1, 1, 0, 1, 1, 1}));

        TensorD<float> x5({3, 3}, TensorInit_Types::RIGHT_HIGHER_NEG_INF);
        assert(x5.vector().equals_to({0, INF_NEG, INF_NEG, 0, 0, INF_NEG, 0, 0, 0}));

        /* disabled as this func is disabled
        x3.deep_copy(x2, 15, 15, {5, 3});
        assert(x3.size() == 15);
        assert(x3.dim()[0] == 5);
        assert(x3._vector->sum() == 15);
        assert(x3.item() == 1);
        assert(x3.dim_to_size() == 15);
        assert(x3.dim_to_size(1) == 3);*/
    }

    static void test_add()
    {
        TensorD<float> x1({1, 2, 3}, TensorInit_Types::Ordinal), x2({1, 3, 3}, TensorInit_Types::Ordinal), y;
        TensorD<float> y_grad, x1_grad, x2_grad;
        x1.vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.vector().set(0, {8, 7, 6, 5, 4, 3, 2, 1, 0});

        x1.add(x2, y, 1, 1, 0, 1, 1);
        assert(y.dim().equals_to({1, 2, 3, 3}));

        assert(y.vector().equals_to({8, 8, 8, 5, 5, 5, 2, 2, 2, 11, 11, 11, 8, 8, 8, 5, 5, 5}));
        y_grad.reset({1, 2, 3, 3}, TensorInit_Types::One);
        y_grad.vector()[0] = 2;
        x1.add_grad(x2, y, y_grad, x1_grad, x2_grad, true, 1, 1, 0, 1, 1);
        assert(x1_grad.vector().equals_to({4, 3, 3, 3, 3, 3}));
        assert(x2_grad.vector().equals_to({3, 2, 2, 2, 2, 2, 2, 2, 2}));

        x1.add(x2, y, 1, 2, 1, 1, 1);
        assert(y.vector().equals_to({8 + 8 + 1, 8 + 7 + 1, 8 + 6 + 1, 5 + 5 + 1, 5 + 4 + 1, 5 + 3 + 1, 2 + 2 + 1, 2 + 1 + 1, 2 + 0 + 1,
                                     11 + 8 + 1, 11 + 7 + 1, 11 + 6 + 1, 8 + 5 + 1, 8 + 4 + 1, 8 + 3 + 1, 5 + 2 + 1, 5 + 1 + 1, 5 + 0 + 1}));

        x1.add(x2, y, 1, 1, 0, 0, 0);
        assert(y.size() == 6 * 9);
        assert(y.dim().equals_to({1, 2, 3, 1, 3, 3}));

        x1.add(x1, y, 1, 1, 0, 0, 1);

        assert(y.vector().equals_to({0, 2, 4, 3, 5, 7, 3, 5, 7, 6, 8, 10}));
        y_grad.reset({1, 2, 1, 2, 3}, TensorInit_Types::One);
        x1_grad.clear();
        x2_grad.clear();
        x1.add_grad(x1, y, y_grad, x1_grad, x2_grad, true, 1, 1, 0, 0, 1);
        assert(x1_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));

        x1.add(x1, y, 1, 1, 0, 2);
        assert(y.vector().equals_to({0, 2, 4, 6, 8, 10}));
        x1.add(x1, y);
        assert(y.vector().equals_to({0, 2, 4, 6, 8, 10}));
        //        x1.add(x1, x1);
        //        assert(y.data().equals_to({0, 2, 4, 6, 8, 10}));
        //        assert(x1.data().equals_to({0, 2, 4, 6, 8, 10}));
    }

    static void test_mul()
    {
        TensorD<float> x1({1, 2, 3}), x2({1, 3, 3}), y;
        TensorD<float> y_grad, x1_grad, x2_grad;
        x1.vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.vector().set(0, {8, 7, 6, 5, 4, 3, 2, 1, 0});

        x1.mul(x2, y, 1, 0, 1, 1);
        assert(y.dim().equals_to({1, 2, 3, 3}));
        assert(y.vector().equals_to({0, 7, 12, 0, 4, 6, 0, 1, 0, 24, 28, 30, 15, 16, 15, 6, 4, 0}));
        y_grad.reset({1, 2, 3, 3}, TensorInit_Types::One);
        y_grad.vector()[0] = 2;
        x1.mul_grad(x2, y, y_grad, x1_grad, x2_grad, true, 1, 0, 1, 1);
        assert(x1_grad.vector().equals_to({8 * 2 + 5 + 2, 7 + 4 + 1, 6 + 3 + 0, 8 + 5 + 2, 7 + 4 + 1, 6 + 3 + 0}));
        assert(x2_grad.vector().equals_to({0 * 2 + 3, 1 + 4, 2 + 5, 0 + 3, 1 + 4, 2 + 5, 0 + 3, 1 + 4, 2 + 5}));
    }

    static void test_dot()
    {
        TensorD<float> x1({1, 2, 3}), x2({1, 3, 3}), y;
        TensorD<float> y_grad, x1_grad, x2_grad;
        x1.vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.vector().set(0, {8, 7, 6, 5, 4, 3, 2, 1, 0});

        x1.dot(x2, y, 1, 1);
        assert(y.dim().equals_to({1, 2, 3}));
        assert(y.vector().equals_to({7 + 12, 4 + 6, 1, 24 + 28 + 30, 15 + 16 + 15, 6 + 4}));
        y_grad.reset({1, 2, 3}, TensorInit_Types::One);
        y_grad.vector()[0] = 2;
        x1.dot_grad(x2, y, y_grad, x1_grad, x2_grad, true, 1, 1);
        assert(x1_grad.vector().equals_to({8 * 2 + 5 + 2, 7 * 2 + 4 + 1, 6 * 2 + 3 + 0, 8 + 5 + 2, 7 + 4 + 1, 6 + 3 + 0}));
        assert(x2_grad.vector().equals_to({0 * 2 + 3, 1 * 2 + 4, 2 * 2 + 5, 0 + 3, 1 + 4, 2 + 5, 0 + 3, 1 + 4, 2 + 5}));
    }

    static void test_mse()
    {
        TensorD<float> x1({1, 2, 3}), x2({3}), y;
        TensorD<float> y_grad, x1_grad, x2_grad;
        x1.vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.vector().set(0, {8, 7, 6});

        x1.mse(x2, y, 0, 1);
        assert(y.dim().equals_to({1, 2}));
        assert(y.vector().equals_to({(64 + 36 + 16) / 3.0, (25 + 9 + 1) / 3.0}));
        y_grad.reset({1, 2}, TensorInit_Types::One);
        x1.mse_grad(x2, y, y_grad, x1_grad, x2_grad, true, 0, 1);
        assert(x1_grad.vector().equals_to({2.0 / 3 * -8, 2.0 / 3 * -6, 2.0 / 3 * -4, 2.0 / 3 * -5, 2.0 / 3 * -3, 2.0 / 3 * -1}));
        assert(x2_grad.vector().equals_to({2.0 / 3 * 13, 2.0 / 3 * 9, 2.0 / 3 * 5}));
    }

    static void test_ce()
    {
        TensorD<float> x1({5}), x2({5}), y, y_grad({1}, TensorInit_Types::One), x1_grad, x2_grad;
        x1.vector().set(0, {0.1, 0.2, 0.4, 0.5, 0.6});
        x2.vector().set(0, {0.2, 0.2, 0.3, 0.4, 0.1});

        x1.ce(x2, y);
        Math<float>::assert_almost_equal(y[0], 4.04548557050879);

        x1.ce_grad(x2, y, y_grad, x1_grad, x2_grad, true);
        // x1_grad all values is log2(x2[j]), x2_grad all values is x1[i] /(x2[i]) / std::log(2)
        assert(x1_grad.vector().equals_to({-std::log2(x2[0]), -std::log2(x2[1]), -std::log2(x2[2]), -std::log2(x2[3]), -std::log2(x2[4])}));
        assert(x2_grad.vector().equals_to({(-x1[0] / x2[0] / std::log(2.0f)), -x1[1] / x2[1] / std::log(2.0f), -x1[2] / x2[2] / std::log(2.0f), -x1[3] / x2[3] / std::log(2.0f), -x1[4] / x2[4] / std::log(2.0f)}));
    }

    static void test_euclidean()
    {
        TensorD<float> x1({4}), x2({4}), y, y_grad({1}, TensorInit_Types::One), x1_grad, x2_grad;
        x1.vector().set(0, {1, 2, 3, 5});
        x2.vector().set(0, {4, 3, 2, 1});

        x1.euclidean(x2, y);
        Math<float>::assert_almost_equal(y[0], std::sqrt(27));

        x1.euclidean_grad(x2, y, y_grad, x1_grad, x2_grad, true);
        // grad: dy_dx1 = 0.5 / y * 2 * (x1[j] - x2[j]), dy_dx2 = -dy_dx1
        assert(x1_grad.vector().equals_to({(x1[0] - x2[0]) / y[0], (x1[1] - x2[1]) / y[0],
                                           (x1[2] - x2[2]) / y[0], (x1[3] - x2[3]) / y[0]}));
        assert(x2_grad.vector().equals_to({-x1_grad[0], -x1_grad[1], -x1_grad[2], -x1_grad[3]}));
    }

    static void test_linear()
    {
        TensorD<float> x1({1, 4}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 4, 9});

        x1.linear(y, 2, 1);
        assert(y.vector().equals_to({1, 3, 9, 19}));

        y_grad.reset({1, 4}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 1, 1, 2});
        x1.linear_grad(y, y_grad, x1_grad, 2, 1);
        assert(x1_grad.vector().equals_to({2, 2, 2, 4}));
    }

    static void test_sqrt()
    {
        TensorD<float> x1({1, 4}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 4, 9});

        x1.sqrt(y);
        assert(y.vector().equals_to({0, 1, 2, 3}));

        y_grad.reset({1, 4}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 1, 1, 2});
        x1.sqrt_grad(y, y_grad, x1_grad);
        assert(x1_grad.vector().equals_to({0, 0.5, 0.25, 1 / 3.0}));
    }

    static void test_pow()
    {
        TensorD<float> x1({1, 4}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 4, 9});

        x1.pow(y, 2);
        assert(y.vector().equals_to({0, 1, 16, 81}));

        y_grad.reset({1, 4}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 1, 1, 2});
        x1.pow_grad(y, y_grad, x1_grad, 2);
        assert(x1_grad.vector().equals_to({0, 2, 8, 36}));
    }

    static void test_softmax()
    {
        TensorD<float> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 2, 3});

        float denominator1 = 1 + std::exp(1);
        float denominator2 = std::exp(2) + std::exp(3);
        x1.softmax(y, 1);
        assert(y.vector().equals_to({std::exp(0.0f) / denominator1, std::exp(1.0f) / denominator1, std::exp(2.0f) / denominator2, std::exp(3.0f) / denominator2}));

        y_grad.reset({2, 2}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 1, 1, 2});
        x1.softmax_grad(y, y_grad, x1_grad, 1);
        //  dL/dx_i = sum(j, dL/dy_j * (i == j - y_j) * y_i) = (dL/dy_i - sum(j, dL/dy_j * y_j)) * y_i
        float temp1 = y[0] + y[1];
        float temp2 = y[2] + 2 * y[3];
        TensorD<float> x1_grad_expected({2, 2});
        x1_grad_expected.vector().set(0, {(1 - temp1) * y[0], (1 - temp1) * y[1], (1 - temp2) * y[2], (2 - temp2) * y[3]});
        assert(x1_grad.vector().equals_to(x1_grad_expected.vector()));
    }

    static void test_activation()
    {
        TensorD<float> x({2, 2}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
        x.activation(Activation_Types::Linear, y);
        assert(y.vector().equals_to(x.vector()));
        y_grad.reset({2, 2}, TensorInit_Types::One);
        x.activation_grad(Activation_Types::Linear, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({1, 1, 1, 1}));

        x.vector()[0] = x.vector()[1] = -1;
        x.activation(Activation_Types::Relu, y);
        assert(y.vector().equals_to({0, 0, 2, 3}));
        x_grad.reset({2, 2}, TensorInit_Types::Zero);
        x.activation_grad(Activation_Types::Relu, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({0, 0, 1, 1}));

        x.reset({2, 2}, TensorInit_Types::Ordinal);
        x.activation(Activation_Types::Sigmoid, y);
        assert(y.vector().equals_to({1.0f / (1.0f + std::exp(0.0f)), 1.0f / (1 + std::exp(-1.0f)), 1.0f / (1 + std::exp(-1.0f * 2)), 1.0f / (1 + std::exp(-1.0f * 3))}));
        x_grad.reset({2, 2}, TensorInit_Types::Zero);
        x.activation_grad(Activation_Types::Sigmoid, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({y[0] * (1 - y[0]), y[1] * (1 - y[1]), y[2] * (1 - y[2]), y[3] * (1 - y[3])}));

        x.reset({2}, TensorInit_Types::Ordinal);
        x.activation(Activation_Types::Tanh, y);
        assert(y.vector().equals_to({2 * Math<float>::sigmoid(0) - 1, Math<float>::tanh(1)}));
        x_grad.reset({2}, TensorInit_Types::Zero);
        y_grad.reset({2}, TensorInit_Types::One);
        x.activation_grad(Activation_Types::Tanh, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({1 - y[0] * y[0], 1 - y[1] * y[1]}));
    }

    static void test_sum()
    {
        TensorD<float> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 2, 3});

        x1.sum(y);
        assert(y.vector().equals_to({6}));

        x1.sum(y, 1);
        assert(y.vector().equals_to({1, 5}));
        y_grad.reset({2}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 2});
        x1.sum_grad(y, y_grad, x1_grad, 1);
        assert(x1_grad.vector().equals_to({1, 1, 2, 2}));

        x1.sum(y, 0);
        assert(y.vector().equals_to({0, 1, 2, 3}));

        x1.sum(y, 2);
        assert(y.vector().equals_to({6}));
    }

    static void test_avg()
    {
        TensorD<float> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 2, 3});

        x1.avg(y);
        assert(y.vector().equals_to({6 / 4.0}));
        y_grad.reset({1}, TensorInit_Types::One);
        y_grad.vector().set(0, {2}, 0, -1);
        x1.avg_grad(y, y_grad, x1_grad);
        assert(x1_grad.vector().equals_to({0.5, 0.5, 0.5, 0.5}));
    }

    static void test_var()
    {
        TensorD<float> x({2, 2}, TensorInit_Types::Ordinal), y, y_grad({2}, TensorInit_Types::One), x_grad;
        x.var(y, false, 1);
        assert(y.dim().equals_to({2}));
        assert(y.vector().equals_to({0.25, 0.25}));
        x.var_grad(y, y_grad, x_grad, false, 1);
        assert(x_grad.vector().equals_to({-0.25, 0.25, -0.25, 0.25}));
    }

    static void test_max()
    {
        TensorD<float> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 2, 1, 3});

        x1.max(y);
        assert(y.vector().equals_to({3}));
        y_grad.reset({1}, TensorInit_Types::One);
        y_grad.vector().set(0, {2}, 0, -1);

        x1.max_grad(y, y_grad, x1_grad);
        assert(x1_grad.vector().equals_to({0, 0, 0, 2}));
    }

    static void test_min()
    {
        TensorD<float> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 2, 1, 3});

        x1.min(y, 1);
        assert(y.vector().equals_to({0, 1}));
        y_grad.reset({2}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 2});

        x1.min_grad(y, y_grad, x1_grad, 1);
        assert(x1_grad.vector().equals_to({1, 0, 2, 0}));
    }

    // swap(1, 3) => [2, 3, 4, 5] => [2, 5, 4, 3]
    // simple example: for one array a[2i][3j][4k][5l], access it by i, l, k, j
    /* input as below: represent as matrix, flatten all the dims >= first_dim + 1
       this is to treat each unit as one elem, and do transpose for this matrix
      01234,56789,*****,*****      ********************
      12345,67890,*****,*****      ********************
      23456,78901,*****,*****      ********************

      yut as below:
      012 567 *** ***
      123 678 *** ***
      ..
      ..
      456 901 *** ***
    */
    static void test_swap()
    {
        TensorD<float> x1({2, 2, 3}), y; // C, H, W
        x1.vector().set(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        x1.swap(y, 0, 1); // H, C, W
        assert(y.vector().equals_to({0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11}));

        x1.swap(y, 0, 2); // W, H, C
        assert(y.vector().equals_to({0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11}));
    }

    /* move_forward({1, 2,3,4,5, 6}, 3,2, 1) => {1, 4, 5, 2, 3, 6}
       move_forward({2,3,4,5,6,7}, 1,2, 4) => {2,5,3,4,6,7} this is equal to move_forward({}, 3,1,1)
       so we will not implement move_backward

       move_forward is a multi_swap actually
    */
    static void test_move_forward()
    {
        TensorD<float> x1({2, 2, 3}), y; // C, H, W
        x1.vector().set(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        x1.move_forward(y, 1, 2, 0); // H, W, C
        assert(y.vector().equals_to({0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11}));

        x1.move_forward(y, 1, 1, 0); // H, C, W
        assert(y.vector().equals_to({0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11}));
    }

    static void test_im2col()
    {
        TensorD<float> x1({1, 2, 4, 4}, TensorInit_Types::Ordinal), y1; // C, H, W
        /*
        channel 0:
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
        channel 1:
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
        */
        x1.im2col(y1, 1, 3, 3, 1, 1, 0, 0);
        assert(y1.vector().equals_to({0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26,
                                      1, 2, 3, 5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23, 25, 26, 27,
                                      4, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 22, 24, 25, 26, 28, 29, 30,
                                      5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23, 25, 26, 27, 29, 30, 31}));

        TensorD<float> x2({1, 2, 4, 4}, TensorInit_Types::Ordinal), y2; // C, H, W
        x2.im2col(y2, 1, 2, 2, 2, 2, 0, 0);
        assert(y2.vector().equals_to({0, 1, 4, 5, 16, 17, 20, 21,
                                      2, 3, 6, 7, 18, 19, 22, 23,
                                      8, 9, 12, 13, 24, 25, 28, 29,
                                      10, 11, 14, 15, 26, 27, 30, 31}));

        /*
        channel 0:
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
        channel 1:
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
        */
        // test padding
        TensorD<float> x3({1, 2, 4, 4}, TensorInit_Types::Ordinal), y3; // C, H, W
        x3.im2col(y3, 1, 3, 3, 3, 3, 1, 1);
        assert(y3.vector().equals_to({0, 0, 0, 0, 0, 1, 0, 4, 5, 0, 0, 0, 0, 16, 17, 0, 20, 21,
                                      0, 0, 0, 2, 3, 0, 6, 7, 0, 0, 0, 0, 18, 19, 0, 22, 23, 0,
                                      0, 8, 9, 0, 12, 13, 0, 0, 0, 0, 24, 25, 0, 28, 29, 0, 0, 0,
                                      10, 11, 0, 14, 15, 0, 0, 0, 0, 26, 27, 0, 30, 31, 0, 0, 0, 0}));

        // test group conv
        // y.reset({groups, batch_size, out_height, out_width, in_channels_per_group, kernel_y, kernel_x});
        TensorD<float> x4({1, 2, 4, 4}, TensorInit_Types::Ordinal), y4; // C, H, W
        x4.im2col(y4, 2, 2, 2, 2, 2, 0, 0);
        assert(y4.vector().equals_to({0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                                      16, 17, 20, 21, 18, 19, 22, 23, 24, 25, 28, 29, 26, 27, 30, 31}));

        // 1*1 kernel
        TensorD<float> x5({1, 2, 4, 4}, TensorInit_Types::Ordinal), y5; // C, H, W
        x5.im2col(y5, 2);
        assert(y5.vector().equals_to(x5.vector()));
    }

    static void test_im2col_grad()
    {
        TensorD<float> x1({1, 2, 4, 4}, TensorInit_Types::Ordinal), y1, y1_grad, x1_grad; // C, H, W
        /*
        channel 0:
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
        channel 1:
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
        */
        x1.im2col(y1, 1, 3, 3, 1, 1, 0, 0);
        y1_grad.reset(y1.dim(), TensorInit_Types::One);
        x1.im2col_grad(y1, y1_grad, x1_grad, 1, 3, 3, 1, 1, 0, 0);
        assert(x1_grad.vector().equals_to({
            1,
            2,
            2,
            1,
            2,
            4,
            4,
            2,
            2,
            4,
            4,
            2,
            1,
            2,
            2,
            1,

            1,
            2,
            2,
            1,
            2,
            4,
            4,
            2,
            2,
            4,
            4,
            2,
            1,
            2,
            2,
            1,
        }));
    }

    static void test_merge_dim()
    {
        TensorD<float> x1({2, 4, 4}, TensorInit_Types::Ordinal), y; // C, H, W
        x1.merge_dim(y, 0, 2);
        assert(y.dim().equals_to({8, 4}));
    }

    static void test_divide()
    {
        TensorD<float> x({2, 3, 2}, TensorInit_Types::Ordinal), x_grad;
        TensorDArray<float> y, y_grad;
        x.divide(y);
        assert(y.size() == 2);
        assert(y[0].dim().equals_to({3, 2}));
        assert(y[0].vector().equals_to({0, 1, 2, 3, 4, 5}));

        assert(y[1].dim().equals_to({3, 2}));
        assert(y[1].vector().equals_to({6, 7, 8, 9, 10, 11}));

        y_grad.reserve(2);
        y_grad[0].reset({3, 2}, TensorInit_Types::One);
        y_grad[1].reset({3, 2}, TensorInit_Types::One);
        x.divide_grad(y, y_grad, x_grad);
        assert(x_grad.dim().equals_to({2, 3, 2}));
        assert(x_grad.vector().equals_to(Vector<float>::One(12)));
    }

    static void test_combine()
    {
        TensorDArray<float> x, x_grad;
        TensorD<float> y, y_grad;
        x.reserve(2);
        x[0].reset({3, 2}, TensorInit_Types::Ordinal);
        x[1].reset({3, 2}, TensorInit_Types::Ordinal);

        y = TensorD<float>::combine(x);
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.vector().equals_to({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
        y_grad.reset({2, 3, 2}, TensorInit_Types::One);
        TensorD<float>::Combine_Grad(x, y, y_grad, x_grad);
        assert(x_grad.size() == 2);
        assert(x_grad[1].dim().equals_to({3, 2}));
        assert(x_grad[1].vector().equals_to(Vector<float>::One(6)));
    }

    static void test_inflate()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
        x.inflate(y, {2});
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.vector().equals_to({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}));

        y_grad.reset({2, 3, 2}, TensorInit_Types::One);
        x.inflate_grad(y, y_grad, x_grad, {2});
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_squeeze()
    {
        TensorD<float> x({1, 2, 1, 3, 4, 1, 1}), y;
        x.squeeze(y);
        assert(y.dim().equals_to({2, 3, 4}));
    }

    static void test_subset()
    {
        TensorD<float> x({3, 2, 2}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
        x.subset(y, {1, 2, 2}, 0);
        assert(y.vector().equals_to({0, 1, 2, 3}));
        assert(y.dim().equals_to({1, 2, 2}));

        y_grad.reset({1, 2, 2}, TensorInit_Types::One);
        x.subset_grad(y, y_grad, x_grad, {1, 2, 2}, 0);
        assert(x_grad.vector().equals_to({1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}));

        x.subset(y, {2, 2, 2}, 4);
        assert(y.vector().equals_to({4, 5, 6, 7, 8, 9, 10, 11}));
        assert(y.dim().equals_to({2, 2, 2}));

        y_grad.reset({2, 2, 2}, TensorInit_Types::One);
        x.subset_grad(y, y_grad, x_grad, {2, 2, 2}, 4);
        assert(x_grad.vector().equals_to({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    }

    static void test_deep_copy()
    {
        TensorD<float> x2({2, 3}, TensorInit_Types::Ordinal), x;
        x.deep_copy(x2, 1);
        assert(x.vector().equals_to({0, 1, 2, 3, 4, 5}));

        x.deep_copy(x2, 2);
        assert(x.vector().equals_to({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
        assert(x.dim().equals_to({2, 2, 3}));
    }

    static void test_equals_to()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        TensorD<float> x1({2, 3}, TensorInit_Types::Ordinal);
        assert(x.equals_to(x1));
        x1.vector()[1] = 100;
        assert(!x.equals_to(x1));
    }

    // {batch_size, node_len}.encode{dict_size, input_dim} => {batch_size, node_len, input_dim} => input encoding
    //    encode(x2, y, 0)

    static void test_encode_by_dict()
    {
        TensorD<float> x({2, 3}), encoder_param({5, 2}), x_encoded;
        x.vector().set(0, {0, 2, 1, 2, 3, 0});
        encoder_param.vector().set(0, {0.1, 0.2, 0.9, 0.2, 0.1, 0.1, 0, 0, 1, 2});
        x.encode_by_dict(encoder_param, x_encoded);
        assert(x_encoded.vector().equals_to({0.1, 0.2, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0, 0, 0.1, 0.2}));

        TensorD<float> x_encoded_grad;
        x_encoded_grad.reset({2, 3, 2}, TensorInit_Types::One);
        TensorD<float> encoder_param_grad;
        x.encode_by_dict_grad(encoder_param, x_encoded, x_encoded_grad, encoder_param_grad);
        assert(encoder_param_grad.dim().equals_to({5, 2}));
        assert(encoder_param_grad.vector().equals_to({2, 2, 1, 1, 2, 2, 1, 1, 0, 0}));
    }

    // {batch_size, node_len}.encode{batch_size, node_len, dict_size} => {batch_size, node_len} => get target prob
    //    encode(x2, y, 2)
    static void test_search_by_dict()
    {
        TensorD<float> target({2, 3}), output({2, 3, 5}), output_decoded;
        target.vector().set(0, {2, 1, 1, 0, 3, 1});
        output.vector().set(0, {0, 0, 0, 1, 0, 0.9, 0.1, 0, 0, 0, 0.2, 0.5, 0.1, 0.0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0});
        target.search_by_dict(output, output_decoded);
        assert(output_decoded.vector().equals_to({0, 0.1, 0.5, 0, 0, 0}));
        assert(output_decoded.dim().equals_to({2, 3}));

        TensorD<float> output_decoded_grad;
        output_decoded_grad.reset({2, 3}, TensorInit_Types::One);
        TensorD<float> output_grad;
        target.search_by_dict_grad(output, output_decoded, output_decoded_grad, output_grad);
        assert(output_grad.dim().equals_to({2, 3, 5}));
        assert(output_grad.vector().equals_to({0, 0, 1, 0, 0,
                                               0, 1, 0, 0, 0,
                                               0, 1, 0, 0, 0,
                                               1, 0, 0, 0, 0,
                                               0, 0, 0, 1, 0,
                                               0, 1, 0, 0, 0}));
    }

    static void test_decode()
    {
        TensorD<float> x({2, 3, 5}), y;
        x.vector().set(0, {0, 0, 0, 1, 0, 0.9, 0.1, 0, 0, 0, 0.2, 0.5, 0.1, 0.0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0});
        x.decode__(y);
        assert(y.vector().equals_to({3, 0, 1, 4, 0, 3}));
        assert(y.dim().equals_to({2, 3}));
    }

    static void test_load()
    {
        TensorD<float> x, y{{2, 3}, TensorInit_Types::Ordinal};
        std::stringstream i;
        i << "start of tensor data\n";
        i << "shape = 2\n";
        i << "2 3\n";
        i << "0 1 2 3 4 5\n";
        i << "end of tensor data\n";
        x.load(i);
        assert(x.equals_to(y));
    }

    static void test_save()
    {
        TensorD<float> x{{2, 3}, TensorInit_Types::Ordinal};
        std::stringstream o;
        x.save(o);

        assert(o.str() == "start of tensor data\nshape = 2\n2 3\n0 1 2 3 4 5\nend of tensor data\n");
    }

    static void test_append()
    {
        TensorD<float> x({2, 2, 2}, TensorInit_Types::Ordinal);
        TensorD<float> x2({2, 2}, TensorInit_Types::Ordinal), y;
        x.append(x2, y);
        assert(y.dim().equals_to({3, 2, 2}));
        assert(y.vector().equals_to({0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3}));

        TensorD<float> x3({2}, TensorInit_Types::Ordinal), y1;
        y.append(x3, y1, 1);
        assert(y1.dim().equals_to({3, 3, 2}));
        assert(y1.vector().equals_to({0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1}));

        TensorD<float> y1_grad({3, 3, 2}, TensorInit_Types::One);
        TensorD<float> y_grad, x3_grad;
        y.append_grad(x3, y1, y1_grad, y_grad, x3_grad, true, 1);
        assert(y_grad.equals_to({3, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
        assert(x3_grad.equals_to({2}, {1, 1}));
    }

    static void test_map()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        x.map(y, [](float v)
              { return v >= 2 ? 1.0f : 0.0f; });
        assert(y.vector().equals_to({0, 0, 1, 1, 1, 1}));

        TensorD<float> y_grad({2, 3}, TensorInit_Types::Ordinal);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.map_grad(y, y_grad, x_grad, [](float v)
                   { return v >= 2 ? 1.0f : 0.0f; });
        assert(x_grad.vector().equals_to({1, 1, 3, 4, 5, 6}));

        TensorD<float> y1;
        x.map(y1, [](const Vector<float> &x, uint start, uint len, Vector<float> &y)
        {
            if (len > 0) y[start] = 1; // <start> node;
            for (uint i = 1; i < len; ++i)
            {
                y[start + i] = x[start + i - 1]; // note: <end> may be skipped
            } 
        }, 1);

        assert(y1.vector().equals_to({1, 0, 1, 1, 3, 4}));


        TensorD<float> y1_grad({2, 3}, TensorInit_Types::Ordinal);
        TensorD<float> x1_grad({2, 3}, TensorInit_Types::One);
        x.map_grad(y1, y1_grad, x1_grad, 
        [](const Vector<float> &x, uint start, uint len, const Vector<float> &y_grad, Vector<float> &x_grad)
        {
            for (uint i = 0; i < len - 1; ++i)
            {
                x_grad[start + i] += y_grad[start + i+1];
            }   
        }, 
        1);

        assert(x1_grad.vector().equals_to({2, 3, 1, 5, 6, 1}));
    }

    static void test_unsqueeze()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        auto y = x.unsqueeze(1);
        assert(y.dim().equals_to({2, 1, 3}));
        assert(y.vector().equals_to({0, 1, 2, 3, 4, 5}));

        TensorD<float> y_grad({2, 1, 3}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.unsqueeze_grad(y, y_grad, x_grad, 1);
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_reshape()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        auto y = x.reshape({3, 2});
        assert(y.dim().equals_to({3, 2}));
        assert(y.vector().equals_to({0, 1, 2, 3, 4, 5}));

        TensorD<float> y_grad({3, 2}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.reshape_grad(y, y_grad, x_grad, {3, 2});
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_where()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        auto y = x.where(CompareTypes::Equal, 2);
        assert(y.size() == 2);
        assert(y[0].dim().equals_to({1}));
        assert(y[0].vector().equals_to({0}));

        assert(y[1].dim().equals_to({1}));
        assert(y[1].vector().equals_to({2}));
    }
    
    static void test_topk()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        auto y = x.topk(2);
        assert(y.size() == 2);
        assert(y[0].dim().equals_to({2, 2}));
        assert(y[0].vector().equals_to({2, 1, 2, 1}));

        assert(y[1].dim().equals_to({2, 2}));
        assert(y[1].vector().equals_to({2, 1, 5, 4}));

        TensorD<float> y1_grad({2, 2}, TensorInit_Types::One);
        TensorD<float> x_grad;
        x.topk_grad(y[0], y1_grad, x_grad, 2);
        assert(x_grad.dim().equals_to({2, 3}));
        assert(x_grad.vector().equals_to({0, 1, 1, 0, 1, 1}));
    }

    static void test_index()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        auto y = x.index({TensorD<float>({2}, {0, 1}), TensorD<float>({2}, {2, 1})}, true);
        assert(y.dim().equals_to({4}));
        assert(y.vector().equals_to({2, 1, 5, 4}));

        TensorD<float> y_grad({4}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.index_grad({TensorD<float>({2}, {0, 1}), TensorD<float>({2}, {2, 1})}, y_grad, x_grad, true);
        assert(x_grad.vector().equals_to({1, 2, 2, 1, 2, 2}));
    }

    static void test_index_non_cross()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        auto y = x.index({TensorD<float>({2}, {0, 1}), TensorD<float>({2}, {2, 1})});
        assert(y.dim().equals_to({2}));
        assert(y.vector().equals_to({2, 4}));

        TensorD<float> y_grad({2}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.index_grad({TensorD<float>({2}, {0, 1}), TensorD<float>({2}, {2, 1})}, y_grad, x_grad);
        assert(x_grad.vector().equals_to({1, 1, 2, 1, 2, 1}));
    }
    static void test_assign()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal);
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        TensorD<float> values({1, 3}, TensorInit_Types::Ordinal);
        TensorD<float> y = x.assign(values, TensorD<float>({1}, {1}));
        assert(y.vector().equals_to({0, 1, 2, 0, 1, 2}));

        TensorD<float> y_grad({2, 3}, TensorInit_Types::One);
        TensorD<float> values_grad({1, 3}, TensorInit_Types::One);
        x.assign_grad(values, TensorD<float>({1}, {1}), y_grad, values_grad);
        assert(values_grad.vector().equals_to({2, 2, 2}));
    }

    static void test_dropout()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        x.vector().set(0, {0, 1, 2, 3, 4, 5});
        x.dropout(y, 0.5f);
        assert(y.dim().equals_to({2, 3}));
        // assert(y.vector().equals_to({0, 1, 2, 3, 4, 5}));

        TensorD<float> y_grad({2, 3}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.dropout_grad(y, y_grad, x_grad, 0.5f);
        // assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_norm_ln()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        x.norm_ln(y, 2, 1);
        assert(y.dim().equals_to({2}));
        assert(y.vector().equals_to({(float)std::sqrt((1 + 2*2.0)/3), (float)std::sqrt((3*3.0 + 4*4 + 5*5)/3)}));

        TensorD<float> y_grad({2}, TensorInit_Types::One), x_grad({2, 3}, TensorInit_Types::One);
        x.norm_ln_grad(y, y_grad, x_grad, 2, 1);
        /*
        ((x1 ^ 2 + x2 ^ 2 + x3 ^ 2) / 3) ^ 0.5
        0.5 / y / len * 2 * x_i => x_i / y / len * y_grad
        */
        assert(x_grad.vector().equals_to({1.0f, 1.0f / y[0] / 3 + 1.0f, 2.0f / y[0] / 3 + 1.0f,
            3.0f / y[1] / 3 + 1.0f, 4.0f / y[1] / 3 + 1.0f, 5.0f / y[1] / 3 + 1.0f}));
    }

    static void test_rms_norm()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        x.rms_norm(y, 2.3f, 1);
        assert(y.dim().equals_to({2, 3}));
        assert(y.vector().equals_to({0, 2.3f * 1.0f / (float)std::sqrt((1 + 2*2.0)/3), 2.3f * 2.0f / (float)std::sqrt((1 + 2*2.0)/3), 2.3f * 3.0f / (float)std::sqrt((3*3.0 + 4*4 + 5*5)/3), 2.3f * 4.0f / (float)std::sqrt((3.0*3 + 4*4 + 5*5)/3), 2.3f *   5.0f / (float)std::sqrt((3*3.0 + 4*4 + 5*5)/3)}));

        TensorD<float> y_grad({2, 3}, TensorInit_Types::One), x_grad({2, 3}, TensorInit_Types::One);
        x.rms_norm_grad(y, y_grad, x_grad, 2.3f, 1);
        assert(x_grad.vector().equals_to({2.78, 1.71, 0.64, 1.16, 1.02, 0.89}, 0.01));
    }

    static void test_rope()
    {
        TensorD<float> x({1, 2, 4}, TensorInit_Types::Ordinal), y;
        x.rope(y);
        assert(y.dim().equals_to({1, 2, 4}));
        assert(y.vector().equals_to({0, 1, 2, 3, -2.05, 6.07, 5.93, 7.06}, 0.01f));

        TensorD<float> y_grad({1, 2, 4}, TensorInit_Types::One), x_grad({1, 2, 4}, TensorInit_Types::One);
        x.rope_grad(y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2.382, 0.699, 2.010, 1.990}, 0.01f));
    }

    static void test_replace()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        y = x.replace(2, 1, 0);
        assert(y.dim().equals_to({2, 3}));
        assert(y.vector().equals_to({0, 0, 1, 0, 0, 0}));

        TensorD<float> y_grad({2, 3}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.replace_grad(y, y_grad, x_grad, 2, 1, 0);
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_insert()
    {
        TensorD<float> x({2, 3}, TensorInit_Types::Ordinal), y;
        y = x.insert(1, 1, 1);
        assert(y.dim().equals_to({2, 3}));
        assert(y.vector().equals_to({0, 1, 1, 3, 1, 4}));
        
        TensorD<float> y_grad({2, 3}, TensorInit_Types::One);
        TensorD<float> x_grad({2, 3}, TensorInit_Types::One);
        x.insert_grad(y, y_grad, x_grad, 1, 1, 1);
        assert(x_grad.vector().equals_to({2, 2, 1, 2, 2, 1}));
    }
};