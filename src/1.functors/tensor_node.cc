#include "../../inc/1.functors/tensor_node.h"
#include "../../inc/1.functors/functor_graph.h"
#include "../../inc/2.operators/fc.h"
#include "../../inc/2.operators/conv.h"
#include "../../inc/2.operators/pooling.h"
#include "../../inc/2.operators/norm.h"
#include "../../inc/2.operators/attentions.h"

Tensor Tensor::Deep_Upgrade(const TensorD<double> &x)
{
    Tensor y(x.dim());
    y.data().deep_copy(x);
    y.grad().reset(y.dim(), TensorInit_Types::Zero);
    return y;
}

Tensor Tensor::Weak_Upgrade(const TensorD<double> &x)
{
    Tensor y(x.dim());
    y.data().weak_copy(x);
    y.grad().reset(y.dim(), TensorInit_Types::Zero);
    return y;
}

TensorList Tensor::Weak_Upgrade(const TensorDArray<double> &x)
{
    TensorList y;
    for (uint i = 0; i < x.size(); ++i)
    {
        y.push_back(Weak_Upgrade(x[i]));
    }

    return y;
}

TensorDArray<double> Tensor::Weak_Data_Downgrade(const TensorList &x)
{
    TensorDArray<double> y;
    for (uint i = 0; i < x.size(); ++i)
    {
        y.push_back(x[i].data());
    }

    return y;
}

void Tensor::Weak_Both_Downgrade(const TensorList &x, TensorDArray<double> &data, TensorDArray<double> &grad)
{
    for (uint i = 0; i < x.size(); ++i)
    {
        data.push_back(x[i].data());
        grad.push_back(x[i].grad());
    }
}

// 0. these are template functions that could call any Functor directly, no need below specific function names
template <typename T, typename... Args>
const Tensor Tensor::f(const Tensor &x2, Args &&...args) const
{
    PFunctor func = std::make_shared<T>(args...);
    TensorList x{*this, x2}, y{Tensor()};
    // LOG_INFO("f: " << func->type());
    func->forward(x, y);
    y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

    if (FunctorGraph::singleton().is_auto_grad() && this->is_auto_grad())
    {
        FunctorGraph::singleton().add(func, x, y);
    }

    return y[0];
}

template <typename T, typename... Args>
Tensor &Tensor::f_(const Tensor &x2, Args &&...args)
{
    PFunctor func = std::make_shared<T>(args...);
    TensorList x{*this, x2}, y{Tensor()};
    // LOG_INFO("f_: " << func->type());
    func->forward(x, y);
    y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

    if (FunctorGraph::singleton().is_auto_grad() && this->is_auto_grad())
    {
        FunctorGraph::singleton().add(func, x, y);
    }

    // put it after _g.add to ensure original this->get() is not released
    this->Ptr<TensorN>::operator=(y[0]);
    return *this;
}

void Tensor::fmm(PFunctor func, const TensorList &x, TensorList &y)
{
    // LOG_INFO("fmm: " << func->type());
    func->forward(x, y);
    assert(x.size() > 0);
    if (FunctorGraph::singleton().is_auto_grad() && x[0].is_auto_grad())
    {
        FunctorGraph::singleton().add(func, x, y);
    }
}

template <typename T, typename... Args>
const Tensor Tensor::f(Args &&...args) const
{
    PFunctor func = std::make_shared<T>(args...);
    TensorList x{*this}, y{Tensor()};

    fmm(func, x, y);
    y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

    return y[0];
}

template <typename T, typename... Args>
Tensor &Tensor::f_(Args &&...args)
{
    PFunctor func = std::make_shared<T>(args...);
    TensorList x{*this}, y{Tensor()};
    fmm(func, x, y);
    y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

    // put it after _g.add to ensure original this->get() is not released
    this->Ptr<TensorN>::operator=(y[0]);
    return *this;
}

template <typename T, typename... Args>
TensorList Tensor::fm(Args &&...args) const
{
    PFunctor func = std::make_shared<T>(args...);
    TensorList x{*this}, y;

    fmm(func, x, y);

    return y;
}

template <typename T, typename... Args>
void Tensor::fmmn(const TensorList &x, TensorList &y, Args &&...args)
{
    PFunctor func = std::make_shared<T>(args...);
    // LOG_INFO("fmmn: " << func->type());
    func->forward(x, y);
}

/*Tensor Tensor::fc(uint input_dim, uint output_dim, bool has_bias, TensorInit_Types w_type, TensorInit_Types b_type) const
{
    Fc fc(input_dim, output_dim, has_bias, w_type, b_type);
    Tensor y;
    fc.forward_first(*this, y);
    return y;
}

Tensor Tensor::conv2d(uint in_channels, uint out_channels, uint kernel_x, uint kernel_y,
                      uint stride_x, uint stride_y, uint padding_x, uint padding_y,
                      uint groups, bool has_bias, TensorInit_Types k_type, TensorInit_Types b_type) const
{
    Conv2d conv2d(in_channels, out_channels, kernel_x, kernel_y, stride_x, stride_y, padding_x, padding_y, groups, has_bias, k_type, b_type);
    Tensor y;
    conv2d.forward_first(*this, y);
    return y;
}

Tensor Tensor::pool2d(Pooling_Types pt, uint kernel_x, uint kernel_y, uint stride_x,
                      uint stride_y, uint padding_x, uint padding_y) const
{
    Pooling2d pool(pt, kernel_x, kernel_y, stride_x, stride_y, padding_x, padding_y);
    Tensor y;
    pool.forward_first(*this, y);
    return y;
}

Tensor Tensor::layer_norm(const Coefficients::LayerNorm& c) const
{
    LayerNorm norm(c);
    Tensor y;
    norm.forward_first(*this, y);
    return y;
}

Tensor Tensor::multi_head_attention(const Coefficients::SelfAttention& c) const
{
    SelfAttention attention(c);
    Tensor y;
    attention.forward_first(*this, y);
    return y;
}*/
