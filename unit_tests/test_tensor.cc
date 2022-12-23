#pragma once
#include "unit_test.h"
#include "../inc/0.tensors/tensor.h"

class TestTensor : public TestClass
{
public:
    REGISTER_TEST_CASES(test_create, test_add, test_mul, test_dot, test_mse, test_ce, test_euclidean, test_linear, test_sqrt, test_pow, test_softmax, test_activation, test_sum, test_avg, test_var, test_max, test_min, test_swap, test_move_forward, test_im2col, test_im2col_grad, test_merge_dim, test_divide, test_combine, test_inflate, test_squeeze, test_subset);

    static void test_create()
    {
        TensorD<double> x1;
        assert(x1.size() == 0);
        assert(x1.shape() == 0);
        TensorD<double> x2({2, 3, 5}, TensorInit_Types::One);
        assert(x2.size() == 30);
        assert(x2.shape() == 3);
        assert(x2.dim()[1] == 3);
        assert(x2.vector().sum() == 30);
        TensorD<double> x3(x2);
        assert(x3._vector->sum() == 30);

        assert(x3.dim_to_size(1) == 15);
        assert(x3.dim_to_size(1, 1) == 3);
        assert(x3.dim_to_size(1, 2, false) == 6);
        assert(x3.size_to_dim(6) == 2);
        assert(x3.size_to_dim(5, false) == 1);
        x3.clear();
        assert(x3.size() == 0);

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
        TensorD<double> x1({1, 2, 3}), x2({1, 3, 3}), y;
        TensorD<double> y_grad, x1_grad, x2_grad;
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
        TensorD<double> x1({1, 2, 3}), x2({1, 3, 3}), y;
        TensorD<double> y_grad, x1_grad, x2_grad;
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
        TensorD<double> x1({1, 2, 3}), x2({1, 3, 3}), y;
        TensorD<double> y_grad, x1_grad, x2_grad;
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
        TensorD<double> x1({1, 2, 3}), x2({3}), y;
        TensorD<double> y_grad, x1_grad, x2_grad;
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
        TensorD<double> x1({5}), x2({5}), y, y_grad({1}, TensorInit_Types::One), x1_grad, x2_grad;
        x1.vector().set(0, {0.1, 0.2, 0.4, 0.5, 0.6});
        x2.vector().set(0, {0.2, 0.2, 0.3, 0.4, 0.1});

        x1.ce(x2, y);
        assert_almost_equal(y[0], 4.04548557050879);

        x1.ce_grad(x2, y, y_grad, x1_grad, x2_grad, true);
        // x1_grad all values is log2(x2[j]), x2_grad all values is x1[i] /(x2[i]) / std::log(2)
        assert(x1_grad.vector().equals_to({-std::log2(x2[0]), -std::log2(x2[1]), -std::log2(x2[2]), -std::log2(x2[3]), -std::log2(x2[4])}));
        assert(x2_grad.vector().equals_to({-x1[0]/x2[0]/std::log(2.0),-x1[1]/x2[1]/std::log(2.0), -x1[2]/x2[2]/std::log(2.0), -x1[3]/x2[3]/std::log(2.0), -x1[4]/x2[4]/std::log(2.0)}));
    }

    static void test_euclidean()
    {
        TensorD<double> x1({4}), x2({4}), y, y_grad({1}, TensorInit_Types::One), x1_grad, x2_grad;
        x1.vector().set(0, {1, 2, 3, 5});
        x2.vector().set(0, {4, 3, 2, 1});

        x1.euclidean(x2, y);
        ALMOST_ZERO(y[0] - std::sqrt(27));

        x1.euclidean_grad(x2, y, y_grad, x1_grad, x2_grad, true);
        // grad: dy_dx1 = 0.5 / y * 2 * (x1[j] - x2[j]), dy_dx2 = -dy_dx1
        assert(x1_grad.vector().equals_to({(x1[0] - x2[0])/y[0], (x1[1] - x2[1]) / y[0], 
            (x1[2] - x2[2])/ y[0], (x1[3] - x2[3])/y[0]}));
        assert(x2_grad.vector().equals_to({-x1_grad[0], -x1_grad[1], -x1_grad[2], -x1_grad[3]}));
    }

    static void test_linear()
    {
        TensorD<double> x1({1, 4}), y, y_grad, x1_grad;
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
        TensorD<double> x1({1, 4}), y, y_grad, x1_grad;
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
        TensorD<double> x1({1, 4}), y, y_grad, x1_grad;
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
        TensorD<double> x1({2, 2}), y, y_grad, x1_grad;
        x1.vector().set(0, {0, 1, 2, 3});

        double denominator1 = 1 + std::exp(1);
        double denominator2 = std::exp(2) + std::exp(3);
        x1.softmax(y, 1);
        assert(y.vector().equals_to({std::exp(0) / denominator1, std::exp(1) / denominator1, std::exp(2) / denominator2, std::exp(3) / denominator2}));

        y_grad.reset({2, 2}, TensorInit_Types::One);
        y_grad.vector().set(0, {1, 1, 1, 2});
        x1.softmax_grad(y, y_grad, x1_grad, 1);
        //  dL/dx_i = sum(j, dL/dy_j * (i == j - y_j) * y_i) = (dL/dy_i - sum(j, dL/dy_j * y_j)) * y_i
        double temp1 = y[0] + y[1];
        double temp2 = y[2] + 2 * y[3];
        TensorD<double> x1_grad_expected({2, 2});
        x1_grad_expected.vector().set(0, {(1 - temp1) * y[0], (1 - temp1) * y[1], (1 - temp2) * y[2], (2 - temp2) * y[3]});
        assert(x1_grad.vector().equals_to(x1_grad_expected.vector()));
    }

    static void test_activation()
    {
        TensorD<double> x({2, 2}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
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
        assert(y.vector().equals_to({1.0 / (1 + std::exp(-1.0 * 0)), 1.0 / (1 + std::exp(-1.0 * 1)), 1.0 / (1 + std::exp(-1.0 * 2)), 1.0 / (1 + std::exp(-1.0 * 3))}));
        x_grad.reset({2, 2}, TensorInit_Types::Zero);
        x.activation_grad(Activation_Types::Sigmoid, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({y[0] * (1 - y[0]), y[1] * (1 - y[1]), y[2] * (1 - y[2]), y[3] * (1 - y[3])}));

        x.reset({2}, TensorInit_Types::Ordinal);
        x.activation(Activation_Types::Tanh, y);
        assert(y.vector().equals_to({2 * Math<double>::sigmoid(0) - 1, Math<double>::tanh(1)}));
        x_grad.reset({2}, TensorInit_Types::Zero);
        y_grad.reset({2}, TensorInit_Types::One);
        x.activation_grad(Activation_Types::Tanh, y, y_grad, x_grad);
        assert(x_grad.vector().equals_to({1 - y[0] * y[0], 1 - y[1] * y[1]}));
    }

    static void test_sum()
    {
        TensorD<double> x1({2, 2}), y, y_grad, x1_grad;
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
        TensorD<double> x1({2, 2}), y, y_grad, x1_grad;
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
        TensorD<double> x({2, 2}, TensorInit_Types::Ordinal), y, y_grad({2}, TensorInit_Types::One), x_grad;
        x.var(y, false, 1);
        assert(y.dim().equals_to({2}));
        assert(y.vector().equals_to({0.25, 0.25}));
        x.var_grad(y, y_grad, x_grad, false, 1);
        assert(x_grad.vector().equals_to({-0.25, 0.25, -0.25, 0.25}));
    }

    static void test_max()
    {
        TensorD<double> x1({2, 2}), y, y_grad, x1_grad;
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
        TensorD<double> x1({2, 2}), y, y_grad, x1_grad;
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
        TensorD<double> x1({2, 2, 3}), y; // C, H, W
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
        TensorD<double> x1({2, 2, 3}), y; // C, H, W
        x1.vector().set(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        x1.move_forward(y, 1, 2, 0); // H, W, C
        assert(y.vector().equals_to({0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11}));

        x1.move_forward(y, 1, 1, 0); // H, C, W
        assert(y.vector().equals_to({0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11}));
    }

    static void test_im2col()
    {
        TensorD<double> x1({1, 2, 4, 4}, TensorInit_Types::Ordinal), y1; // C, H, W
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

        TensorD<double> x2({1, 2, 4, 4}, TensorInit_Types::Ordinal), y2; // C, H, W
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
        TensorD<double> x3({1, 2, 4, 4}, TensorInit_Types::Ordinal), y3; // C, H, W
        x3.im2col(y3, 1, 3, 3, 3, 3, 1, 1);
        assert(y3.vector().equals_to({0, 0, 0, 0, 0, 1, 0, 4, 5, 0, 0, 0, 0, 16, 17, 0, 20, 21,
                                      0, 0, 0, 2, 3, 0, 6, 7, 0, 0, 0, 0, 18, 19, 0, 22, 23, 0,
                                      0, 8, 9, 0, 12, 13, 0, 0, 0, 0, 24, 25, 0, 28, 29, 0, 0, 0,
                                      10, 11, 0, 14, 15, 0, 0, 0, 0, 26, 27, 0, 30, 31, 0, 0, 0, 0}));

        // test group conv
        // y.reset({groups, batch_size, out_height, out_width, in_channels_per_group, kernel_y, kernel_x});
        TensorD<double> x4({1, 2, 4, 4}, TensorInit_Types::Ordinal), y4; // C, H, W
        x4.im2col(y4, 2, 2, 2, 2, 2, 0, 0);
        assert(y4.vector().equals_to({0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                                      16, 17, 20, 21, 18, 19, 22, 23, 24, 25, 28, 29, 26, 27, 30, 31}));

        // 1*1 kernel
        TensorD<double> x5({1, 2, 4, 4}, TensorInit_Types::Ordinal), y5; // C, H, W
        x5.im2col(y5, 2);
        assert(y5.vector().equals_to(x5.vector()));
    }

    static void test_im2col_grad()
    {
        TensorD<double> x1({1, 2, 4, 4}, TensorInit_Types::Ordinal), y1, y1_grad, x1_grad; // C, H, W
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
        TensorD<double> x1({2, 4, 4}, TensorInit_Types::Ordinal), y; // C, H, W
        x1.merge_dim(y, 0, 2);
        assert(y.dim().equals_to({8, 4}));
    }

    static void test_divide()
    {
        TensorD<double> x({2, 3, 2}, TensorInit_Types::Ordinal), x_grad;
        TensorDArray<double> y, y_grad;
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
        assert(x_grad.vector().equals_to(Vector<double>::one(12)));
    }

    static void test_combine()
    {
        TensorDArray<double> x, x_grad;
        TensorD<double> y, y_grad;
        x.reserve(2);
        x[0].reset({3, 2}, TensorInit_Types::Ordinal);
        x[1].reset({3, 2}, TensorInit_Types::Ordinal);

        y = TensorD<double>::combine(x);
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.vector().equals_to({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
        y_grad.reset({2, 3, 2}, TensorInit_Types::One);
        TensorD<double>::combine_grad(x, y, y_grad, x_grad);
        assert(x_grad.size() == 2);
        assert(x_grad[1].dim().equals_to({3, 2}));
        assert(x_grad[1].vector().equals_to(Vector<double>::one(6)));
    }

    static void test_inflate()
    {
        TensorD<double> x({2, 3}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
        x.inflate(y, {2});
        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.vector().equals_to({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}));

        y_grad.reset({2, 3, 2}, TensorInit_Types::One);
        x.inflate_grad(y, y_grad, x_grad, {2});
        assert(x_grad.vector().equals_to({2, 2, 2, 2, 2, 2}));
    }

    static void test_squeeze()
    {
        TensorD<double> x({1, 2, 1, 3, 4, 1, 1}), y;
        x.squeeze(y);
        assert(y.dim().equals_to({2, 3, 4}));
    }

    static void test_subset()
    {
        TensorD<double> x({3, 2, 2}, TensorInit_Types::Ordinal), y, y_grad, x_grad;
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
};