#ifndef ENABLE_CUDA

#include "inc/0.tensors/predefs.h"
#include "inc/0.tensors/cuda_ctx.h"
#include <iostream>
#include <chrono>
#include <float.h>

template <class T>
float* CudaCtx<T>::alloc_device(size_t size)
{
    return nullptr;
}

template <class T>
void CudaCtx<T>::free_device(float *device)
{
}

template <class T>
float* CudaCtx<T>::to_device(const T* h, size_t size)
{
    return nullptr;
}

template <class T>
void CudaCtx<T>::to_host(float* h, const float* d, size_t size)
{
}

template <class T>
bool CudaCtx<T>::runnable() 
{ 
    return false;
}

template <class T>
 void CudaCtx<T>::add(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1, float alpha_x2, float beta)
 {

 }

template <class T>
void CudaCtx<T>::add_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha_x1, float alpha_x2, float beta)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::mul(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha, float beta)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::mul_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols, float alpha, float beta)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::dot(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::dot_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::mse(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::mse_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::ce(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::ce_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::euclidean(const float* x1, const float* x2, float* y, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::euclidean_grad(const float* x1, const float* x2, const float* y_grad, float* x1_grad, float* x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::linear(const float* x, float* y, uint rows, uint cols, float alpha, float beta)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::linear_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float alpha, float beta)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::sqrt(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::sqrt_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::pow(const float* x, float* y, uint rows, uint cols, float n, float bias)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::pow_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float n, float bias)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::ln(const float* x, float* y, uint rows, uint cols, float bias)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::ln_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float bias)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::softmax(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::softmax_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::activation(const float* x, float* y, uint rows, uint cols, Activation_Types type)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::activation_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, Activation_Types type)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::rms_norm(const float* x, float* y, uint rows, uint cols, float gamma)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::rms_norm_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols, float gamma)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::sum(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::sum_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::avg(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::avg_grad(const float* x, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::max(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::max_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::min(const float* x, float* y, uint rows, uint cols)
{
    // 空实现
}

template <class T>
void CudaCtx<T>::min_grad(const float* x, const float* y, const float* y_grad, float* x1_grad, uint rows, uint cols)
{
    // 空实现
}

template class CudaCtx<float>;
#endif


