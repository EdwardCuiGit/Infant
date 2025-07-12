#include <cuda_runtime.h>
#include "inc/0.tensors/common_prefix.h"

__global__ void printVariable(float* d_value) {
    printf("Value from GPU: %f\n", *d_value);
}

struct BinaryOffset
{
    uint x1_start;
    uint x2_start;
    uint y_map_start;
    uint y_reduce_start;
};

struct UnaryOffset
{
    uint row;
    uint x_start;
    uint y_map_start;
    uint y_reduce_start;
};

__device__ BinaryOffset _get_offset(uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o;
    uint row_x1 = blockIdx.x * blockDim.x + threadIdx.x;
    uint row_x2 = blockIdx.y * blockDim.y + threadIdx.y;
    uint depth = blockIdx.z * blockDim.z + threadIdx.z;

    o.y_map_start = depth * rows_x1 * rows_x2 * cols + row_x1 * rows_x2 * cols + row_x2 * cols;
    o.y_reduce_start = depth * rows_x1 * rows_x2 + row_x1 * rows_x2 + row_x2;
    o.x1_start = depth * rows_x1 * cols + row_x1 * cols;
    o.x2_start = depth * rows_x2 * cols + row_x2 * cols;
    return o;
}

__device__ UnaryOffset _get_offset(uint rows, uint cols)
{
    UnaryOffset o;
    o.row = blockIdx.z * blockDim.x + threadIdx.x;
    o.x_start = o.row * cols;
    o.y_map_start = o.row * cols;
    o.y_reduce_start = o.row;
    return o;
}

__global__ void add_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1 = 1.0f, float alpha_x2 = 1.0f, float beta = 0.0f)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        y[o.y_map_start + i] = x1[o.x1_start + i] * alpha_x1 + x2[o.x2_start + i] * alpha_x2 + beta;
    }
}

__global__ void add_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1 = 1.0f, float alpha_x2 = 1.0f, float beta = 0.0f)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        // note: use += to accumulate grads from different functors
        float grad1 = y_grad[o.y_map_start + i] * alpha_x1;
        float grad2 = y_grad[o.y_map_start + i] * alpha_x2;
        atomicAdd(&x1_grad[o.x1_start + i], grad1);
        atomicAdd(&x2_grad[o.x2_start + i], grad2);
    }
}

__global__ void mul_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha = 1.0f, float beta = 0.0f)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        y[o.y_map_start + i] = x1[o.x1_start + i] * x2[o.x2_start + i] * alpha + beta;
    }
}

__global__ void mul_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha = 1.0f, float beta = 0.0f)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        atomicAdd(&x1_grad[o.x1_start + i], y_grad[o.y_map_start + i] * x2[o.x2_start + i] * alpha);
        atomicAdd(&x2_grad[o.x2_start + i], y_grad[o.y_map_start + i] * x1[o.x1_start + i] * alpha);
    }
}

__global__ void dot_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    y[o.y_reduce_start] = 0;
    for (int i = 0; i < cols; ++i)
    {
        y[o.y_reduce_start] += x1[o.x1_start + i] * x2[o.x2_start + i];
    }
}

__global__ void dot_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        atomicAdd(&x1_grad[o.x1_start + i], y_grad[o.y_reduce_start] * x2[o.x2_start + i]);
        atomicAdd(&x2_grad[o.x2_start + i], y_grad[o.y_reduce_start] * x1[o.x1_start + i]);
    }
}

__global__ void mse_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    y[o.y_reduce_start] = 0;

    for (int i = 0; i < cols; ++i)
    {
        float va = x1[o.x1_start + i];
        float vb = x2[o.x2_start + i];
        y[o.y_reduce_start] += (va - vb) * (va - vb) / cols;
    }
}

__global__ void mse_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        float grad = y_grad[o.y_reduce_start] * (x1[o.x1_start + i] - x2[o.x2_start + i]) * 2 / cols;
        atomicAdd(&x1_grad[o.x1_start + i], grad);
        atomicAdd(&x2_grad[o.x2_start + i], -1 * grad);
    }
}

__global__ void ce_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    y[o.y_reduce_start] = 0;

    for (int i = 0; i < cols; ++i)
    {
        float va = x1[o.x1_start + i];
        float vb = x2[o.x2_start + i];
        y[o.y_reduce_start] += -1.0f * va * log2f(vb + 1e-10f);
    }
}

__global__ void ce_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    for (int i = 0; i < cols; ++i)
    {
        float grad = y_grad[o.y_reduce_start] * -1.0f * log2f(x2[o.x2_start + i] + 1e-10);
        atomicAdd(&x1_grad[o.x1_start + i], grad);
        atomicAdd(&x2_grad[o.x2_start + i], -1 * x1[o.x1_start + i] / (x2[o.x2_start + i] + 1e-10) / logf(2.0f));
    }
}

__global__ void euclidean_kernel(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);

    y[o.y_reduce_start] = 0;

    for (int i = 0; i < cols; ++i)
    {
        float va = x1[o.x1_start + i];
        float vb = x2[o.x2_start + i];
        y[o.y_reduce_start] += (va - vb) * (va - vb);
    }

    y[o.y_reduce_start] = sqrtf(y[o.y_reduce_start]);
}

__global__ void euclidean_grad_kernel(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    BinaryOffset o = _get_offset(depths, rows_x1, rows_x2, cols);


    // recalculate y
    float y = 0;
    for (int i = 0; i < cols; ++i)
    {
        float va = x1[o.x1_start + i];
        float vb = x2[o.x2_start + i];
        y += (va - vb) * (va - vb);
    }

    y = sqrtf(y);

    if (y != 0)
    {
        for (int i = 0; i < cols; ++i)
        {
            float grad = y_grad[o.y_reduce_start] * (x1[o.x1_start + i] - x2[o.x2_start + i]) / y;
            atomicAdd(&x1_grad[o.x1_start + i], grad);
            atomicAdd(&x2_grad[o.x2_start + i], -1 * grad);
        }
    }
}

__global__ void linear_kernel(const float* x, float* y, uint rows, uint cols, float alpha, float beta) 
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = x[o.x_start + col] * alpha + beta;
    }
}

__global__ void linear_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float alpha, float beta)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_map_start + col] * alpha);
    }
}

__global__ void sqrt_kernel(const float* x, float* y, uint rows, uint cols)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = sqrtf(x[o.x_start + col]);
    }
}

__global__ void sqrt_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        float grad = x[o.x_start + col] == 0 ? 0 : 0.5f / sqrtf(x[o.x_start + col]);
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_map_start + col] * grad);
    }
}

__global__ void pow_kernel(const float* x, float* y, uint rows, uint cols, float n, float bias)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = powf(x[o.x_start + col] + bias, n);
    }
}

__global__ void pow_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float n, float bias)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_map_start + col] * n * powf(x[o.x_start + col] + bias, n - 1));
    }
}

__global__ void ln_kernel(const float* x, float* y, uint rows, uint cols, float bias)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = logf(x[o.x_start + col] + bias);
    }
}

__global__ void ln_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float bias)
{
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_map_start + col] / (x[o.x_start + col] + bias));
    }
}

__global__ void softmax_kernel(const float* x, float* y, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float max_val = -FLT_MAX;
    for (int col = 0; col < cols; col++) {
        float val = x[o.x_start + col];
        max_val = max(max_val, val);
    }

    if (max_val == -FLT_MAX) {
        for (int col = 0; col < cols; col++) {
            y[o.y_map_start + col] = 0;
        }
        return;
    }

    float sum = 0.0f;
    
    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = expf(x[o.x_start + col] - max_val);
        sum += y[o.y_map_start + col];
    }

    if (sum != 0.0f)
    {
        for (int col = 0; col < cols; col++) {
            y[o.y_map_start + col] /= sum;
        }
    }
}

__global__ void softmax_grad_kernel(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float sum = 0;
    for (int i = 0; i < cols; ++i)
    {
        sum += y[o.y_map_start + i] * y_grad[o.y_map_start + i];
    }

    for (int col = 0; col < cols; col++) {
        float grad = y[o.y_map_start + col] * (y_grad[o.y_map_start + col] - sum);
        atomicAdd(&x1_grad[o.x_start + col], grad);
    }
}

__global__ void activation_kernel(const float* x, float* y, uint rows, uint cols, Activation_Types type) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        float val = x[o.x_start + col];
        float res;
        switch (type) {
            case Activation_Types::None:
                res = val;
                break;
            case Activation_Types::Linear:
                res = val;
                break;
            case Activation_Types::Relu:
                res = val > 0 ? val : 0;
                break;
            case Activation_Types::Tanh:
                res = tanhf(val);
                break;
            case Activation_Types::Sigmoid:
                res = 1.0f / (1.0f + expf(-val));
                break;
            default:
                res = val;
                break;
        }

        y[o.y_map_start + col] = res;
    }
}

// note: can add y
__global__ void activation_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, Activation_Types type) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        float grad = 0;
        switch (type) {
            case Activation_Types::None:
                grad = 1;
                break;
            case Activation_Types::Linear:
                grad = 1;
                break;
            case Activation_Types::Relu:
                grad = x[o.x_start + col] > 0 ? 1 : 0;
                break;
            case Activation_Types::Tanh:
                grad = 1 - tanhf(x[o.x_start + col]) * tanhf(x[o.x_start + col]);
                break;
            case Activation_Types::Sigmoid:
                float sigmoid_val = 1.0f / (1.0f + expf(-x[o.x_start + col]));
                grad = sigmoid_val * (1 - sigmoid_val);
                break;
            default:
                grad = 1;
                break;
        }

        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_map_start + col] * grad);
    }
}

__global__ void rms_norm_kernel(const float* x, float* y, uint rows, uint cols, float gamma) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += x[o.x_start + col] * x[o.x_start + col];
    }

    float norm = sqrtf(sum / cols);

    for (int col = 0; col < cols; col++) {
        y[o.y_map_start + col] = x[o.x_start + col] / (norm + 1e-10) * gamma;
    }
}

__global__ void rms_norm_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float gamma) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows || cols == 0) return;

    float sum = 0.0f;
    float grad_sum = 0;
    for (int col = 0; col < cols; col++) {
        sum += x[o.x_start + col] * x[o.x_start + col];
        grad_sum += y_grad[o.y_map_start + col] * x[o.x_start + col];
    }

    float norm = sqrtf(sum / cols);

    grad_sum /= (norm + 1e-10) * (norm + 1e-10) * (norm + 1e-10) * cols;


    for (int col = 0; col < cols; col++) {
        float grad = gamma * y_grad[o.y_map_start + col] / (norm + 1e-10) - gamma * grad_sum * x[o.x_start + col];
        atomicAdd(&x1_grad[o.x_start + col], grad);
    }
}

__global__ void sum_kernel(const float* x, float* y, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += x[o.x_start + col];
    }

    y[o.y_reduce_start] = sum;
}

__global__ void sum_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_reduce_start]);
    }
}

__global__ void avg_kernel(const float* x, float* y, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += x[o.x_start + col];
    }

    y[o.y_reduce_start] = sum / cols;
}

__global__ void avg_grad_kernel(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_reduce_start] / cols);
    }
}

__global__ void max_kernel(const float* x, float* y, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float max_val = -FLT_MAX;
    for (int col = 0; col < cols; col++) {
        float val = x[o.x_start + col];
        max_val = max(max_val, val);
    }

    y[o.y_reduce_start] = max_val;
}

__global__ void max_grad_kernel(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_reduce_start] * (x[o.x_start + col] == y[o.y_reduce_start]));
    }
}

__global__ void min_kernel(const float* x, float* y, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    float min_val = FLT_MAX;
    for (int col = 0; col < cols; col++) {
        float val = x[o.x_start + col];
        min_val = min(min_val, val);
    }

    y[o.y_reduce_start] = min_val;
}

__global__ void min_grad_kernel(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols) {
    UnaryOffset o = _get_offset(rows, cols);

    if (o.row >= rows) return;

    for (int col = 0; col < cols; col++) {
        atomicAdd(&x1_grad[o.x_start + col], y_grad[o.y_reduce_start] * (x[o.x_start + col] == y[o.y_reduce_start]));
    }
}







