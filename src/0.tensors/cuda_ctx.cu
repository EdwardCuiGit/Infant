#ifdef ENABLE_CUDA

#include "inc/0.tensors/predefs.h"
#include "inc/0.tensors/cuda_ctx.h"
#include <iostream>
#include <chrono>
#include <float.h>

#include "cuda_kernels.cu"

template <class T>
float* CudaCtx<T>::alloc_device(size_t size)
{
    float* d;
    cudaMalloc((void**)&d, size * sizeof(float));
    return d;
}

template <class T>
void CudaCtx<T>::free_device(float *device)
{
    cudaFree(device);
}

template <class T>
float* CudaCtx<T>::to_device(const T* h, size_t size)
{
    float* d = alloc_device(size);
    cudaMemcpy(d, h, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return d;
}

template <class T>
void CudaCtx<T>::to_host(float* h, const float* d, size_t size)
{
    cudaMemcpy(h, d, size * sizeof(float), cudaMemcpyDeviceToHost);
}

template <class T>
bool CudaCtx<T>::runnable() 
{ 
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return Environment::Enabled_Cuda() && deviceCount > 0;
}

void _get_dims(uint depths, uint rows_x1, uint rows_x2, uint cols, dim3& blocks, dim3& threads)
{
    uint threads_per_block = 1;
    uint blocks_x = (rows_x1 + threads_per_block - 1) / threads_per_block;
    uint blocks_y = (rows_x2 + threads_per_block - 1) / threads_per_block;
    uint blocks_z = (depths + threads_per_block - 1) / threads_per_block;
    blocks = dim3(blocks_x, blocks_y, blocks_z);
    threads = dim3(threads_per_block, threads_per_block, threads_per_block);
}
// binary map
template <class T>
void CudaCtx<T>::add(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1, float alpha_x2, float beta)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    // std::cout << "add" << std::endl;
    // std::cout << blocks_x << " " << blocks_y << " " << blocks_z << std::endl;
    // std::cout << rows_a << " " << rows_b << " " << depths << " " << cols<< std::endl;

    add_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols, alpha_x1, alpha_x2, beta);
}

template <class T>
void CudaCtx<T>::add_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1, float alpha_x2, float beta)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    add_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols, alpha_x1, alpha_x2, beta);
}

template <class T>
void CudaCtx<T>::mul(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha, float beta)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    mul_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols, alpha, beta);
}

template <class T>
void CudaCtx<T>::mul_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha, float beta)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    mul_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols, alpha, beta);
}

// binary reduce 
template <class T>
void CudaCtx<T>::dot(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    dot_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::dot_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    dot_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::mse(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    mse_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::mse_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    mse_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::ce(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    ce_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::ce_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    ce_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::euclidean(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    euclidean_kernel<<<blocks, threads>>>(x1, x2, y, depths, rows_x1, rows_x2, cols);
}

template <class T>
void CudaCtx<T>::euclidean_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    dim3 blocks, threads;
    _get_dims(depths, rows_x1, rows_x2, cols, blocks, threads);

    euclidean_grad_kernel<<<blocks, threads>>>(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols);
}

void _get_dims(uint rows, uint cols, uint& blocks, uint& threads_per_block)
{
    threads_per_block = 64;
    blocks = (rows + threads_per_block - 1) / threads_per_block;
}

// unary map
template <class T>
void CudaCtx<T>::linear(const float* x, float* y, uint rows, uint cols, float alpha, float beta)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    linear_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols, alpha, beta);
}

template <class T>
void CudaCtx<T>::linear_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float alpha, float beta)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    linear_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols, alpha, beta);
}

template <class T>
void CudaCtx<T>::sqrt(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    sqrt_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::sqrt_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    sqrt_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols);
}

template <class T>
void CudaCtx<T>::pow(const float* x, float* y, uint rows, uint cols, float n, float bias)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    pow_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols, n, bias);
}

template <class T>
void CudaCtx<T>::pow_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float n, float bias)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    pow_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols, n, bias);
}


template <class T>
void CudaCtx<T>::ln(const float* x, float* y, uint rows, uint cols, float bias)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    ln_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols, bias);
}

template <class T>
void CudaCtx<T>::ln_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float bias)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    ln_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols, bias);
}

template <class T>
void CudaCtx<T>::softmax(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    softmax_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::softmax_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    softmax_grad_kernel<<<blocks, threads_per_block>>>(x, y, y_grad, x1_grad, rows, cols);
}

template <class T>
void CudaCtx<T>::activation(const float* x, float* y, uint rows, uint cols, Activation_Types type)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    activation_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols, type);
}

template <class T>
void CudaCtx<T>::activation_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, Activation_Types type)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    activation_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols, type);
}

template <class T>
void CudaCtx<T>::rms_norm(const float* x, float* y, uint rows, uint cols, float gamma)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    rms_norm_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols, gamma);
}

template <class T>
void CudaCtx<T>::rms_norm_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float gamma)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    rms_norm_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols, gamma);
}

// unary reduce 
template <class T>
void CudaCtx<T>::sum(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    sum_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::sum_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    sum_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols);
}

template <class T>
void CudaCtx<T>::avg(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    avg_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::avg_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    avg_grad_kernel<<<blocks, threads_per_block>>>(x, y_grad, x1_grad, rows, cols);
}

template <class T>
void CudaCtx<T>::max(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    max_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::max_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    max_grad_kernel<<<blocks, threads_per_block>>>(x, y, y_grad, x1_grad, rows, cols);
}

template <class T>
void CudaCtx<T>::min(const float* x, float* y, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    min_kernel<<<blocks, threads_per_block>>>(x, y, rows, cols);
}

template <class T>
void CudaCtx<T>::min_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    uint blocks, threads_per_block;
    _get_dims(rows, cols, blocks, threads_per_block);

    min_grad_kernel<<<blocks, threads_per_block>>>(x, y, y_grad, x1_grad, rows, cols);
}

template class CudaCtx<float>;

#endif

