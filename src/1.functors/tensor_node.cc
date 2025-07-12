#include "inc/1.functors/tensor_node.h"

Tensor Tensor::Deep_Upgrade(const TensorD<float> &x)
{
    Tensor y(x.dim());
    y.data().deep_copy(x);
    y.grad().reset(y.dim(), TensorInit_Types::Zero);
    return y;
}

Tensor Tensor::Weak_Upgrade(const TensorD<float> &x)
{
    Tensor y(x.dim());
    y.data().weak_copy(x);
    y.grad().reset(y.dim(), TensorInit_Types::Zero);
    return y;
}

TensorList Tensor::Weak_Upgrade(const TensorDArray<float> &x)
{
    TensorList y;
    for (uint i = 0; i < x.size(); ++i)
    {
        y.push_back(Weak_Upgrade(x[i]));
    }

    return y;
}

TensorDArray<float> Tensor::Weak_Data_Downgrade(const TensorList &x)
{
    TensorDArray<float> y;
    for (uint i = 0; i < x.size(); ++i)
    {
        y.push_back(x[i].data());
    }

    return y;
}

void Tensor::Weak_Both_Downgrade(const TensorList &x, TensorDArray<float> &data, TensorDArray<float> &grad)
{
    for (uint i = 0; i < x.size(); ++i)
    {
        data.push_back(x[i].data());
        grad.push_back(x[i].grad());
    }
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
