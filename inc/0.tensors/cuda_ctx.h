#pragma once

#include "common_prefix.h"

template <class T>
class CudaCtx
{
public:
    static float* alloc_device(size_t size);

    static void free_device(float *device);

    static float* to_device(const T* h, size_t size);

    static void to_host(float* h, const float* d, size_t size);

    static bool runnable();

    // binary map: add, mul
    static void add(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1 = 1.0, float alpha_x2 = 1.0, float beta = 1.0);
    static void add_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1 = 1.0, float alpha_x2 = 1.0, float beta = 1.0);
    static void mul(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha = 1.0, float beta = 1.0);
    static void mul_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha = 1.0, float beta = 1.0);

    // binary reduce: dot, mse, ce, euclidean
    static void dot(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void dot_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void mse(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void mse_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void ce(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void ce_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void euclidean(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols);
    static void euclidean_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols);

    // unary map: linear, sqrt, pow, ln, softmax, activation, rms_norm
    static void linear(const float* x, float* y, uint rows, uint cols, float alpha = 1.0, float beta = 0.0);
    static void linear_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float alpha = 1.0, float beta = 0.0);
    static void sqrt(const float* x, float* y, uint rows, uint cols);
    static void sqrt_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols);
    static void pow(const float* x, float* y, uint rows, uint cols, float n, float bias = 0);
    static void pow_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float n, float bias = 0);
    static void ln(const float* x, float* y, uint rows, uint cols, float bias = 0);
    static void ln_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float bias = 0);
    static void softmax(const float* x, float* y, uint rows, uint cols);
    static void softmax_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols);
    static void activation(const float* x, float* y, uint rows, uint cols, Activation_Types type);
    static void activation_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, Activation_Types type);
    static void rms_norm(const float* x, float* y, uint rows, uint cols, float gamma);
    static void rms_norm_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float gamma);

    // unary reduce: sum, avg, max, min
    static void sum(const float* x, float* y, uint rows, uint cols);
    static void sum_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols);
    static void avg(const float* x, float* y, uint rows, uint cols);
    static void avg_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols);
    static void max(const float* x, float* y, uint rows, uint cols);
    static void max_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols);
    static void min(const float* x, float* y, uint rows, uint cols);
    static void min_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols);

    // data manipulation funcs needed for transformer: rope, swap, move_forward, combine, subset, topk, where, index, assign, append, decode__, insert, encode_by_dict, replace
};
