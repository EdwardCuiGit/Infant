#pragma once

#include "operator_base.h"
/*
Convolution operators:
1) Conv2d
2) Conv3d, TODO
3) deconv
*/

// projection op: tensor<N, C, H, W> -> tensor<N, C1, H1, W1>
/*Conv2d:
    https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
    supports padding, 
        TODO: padding_mode
    supports grouped: less computes
        TODO: shuffled grouped conv
        TODO: pointwise grouped conv
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    TODO: transposed conv; also call deconv, even not accurate
    TODO: dilated/Atrous:  larger receptive field
        https://arxiv.org/abs/1412.7062
        when d = 2, each pixel's left/top/right/bottom has one distance to the other
    TODO: separable conv
        TODO: spatially separable conv
        TODO: DepthwiseConv2d: groups == in_channels
    TODO: flatten conv
        https://arxiv.org/abs/1412.5474
    TODO: deconv
    sugar classes:
        Conv2d1x1: stride == 1, padding == 0, dilation == 1
        TODO: Conv2d3x3: stride == 3
*/
// note: not supporting config & param save/load
class Conv2d : public UnOp
{
private:
    uint _in_channels, _out_channels, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y,
        _groups, _in_channels_per_group, _out_channels_per_group;
    bool _has_bias;

    Tensor _k, _b; // parameters for kernel

public:
    // padding is on 4 sides
    // TODO: dilation is not used
    Conv2d(uint in_channels, uint out_channels, uint kernel_x = 3, uint kernel_y = 3, uint stride_x = 1, uint stride_y = 1,
           uint padding_x = 0, uint padding_y = 0, /*uint dilation_x = 1, uint dilation_y = 1,*/ uint groups = 1, bool has_bias = true,
           TensorInit_Types k_type = TensorInit_Types::Gaussian, TensorInit_Types b_type = TensorInit_Types::Zero)
        : UnOp(nullptr, "Conv2d"), _in_channels(in_channels), _out_channels(out_channels), _kernel_x(kernel_x), _kernel_y(kernel_y),
          _stride_x(stride_x), _stride_y(stride_y), _padding_x(padding_x), _padding_y(padding_y),
          /*_dilation_x(dilation_x), _dilation_y(dilation_y),*/ _groups(groups), _has_bias(has_bias)
    {
        assert(in_channels > 0 && out_channels > 0);
        assert(kernel_x > 0 && kernel_y > 0);
        assert(stride_x > 0 && stride_y > 0);
        assert(groups > 0 && in_channels % groups == 0);
        _in_channels_per_group = in_channels / groups;

        assert(out_channels % groups == 0);
        _out_channels_per_group = out_channels / groups;

        _k = add_param("k", {groups, _out_channels_per_group, _in_channels_per_group, kernel_y, kernel_x}, k_type);
        if (has_bias)
        {
            _b = add_param("b", {_out_channels_per_group}, b_type);
        }
    }

    // x: [N:batch_size, C:in_channels, H:in_height, W:in_width]
    // y: [N:batch_size, C:out_channels, H: out_height, W:out_width]
    virtual Tensor forward(const Tensor &x) const override
    {
        assert(x.shape() >= 4);
        uint batch_size = x.dim()[0], in_channels = x.dim()[1], in_height = x.dim()[2], in_width = x.dim()[3];
        assert(in_channels == _in_channels);

        // grouped conv
        // col: [groups, batch_size, out_height, out_width, in_channels_per_group, kernel_y, kernel_x]
        Tensor col = x.im2col(_groups, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y);
        // k: [groups, out_channels_per_group, in_channels_per_group, kernel_y, kernel_x]
        // use last 3 dim [in_channels_per_group, kernel_y, kernel_x] as one unit for dot
        // use groups as matching dim from col & k, i.e., only matched dim will generate dot
        Tensor y = col.dot(_k, 1, 3);
        // note: not using below because harder to add bias
        // _k->dot(y, 1, 3); // [groups, out_channels_per_group, batch_size, out_height, out_width];
        // y: [groups, batch_size, out_height, out_width, out_channels_per_group]
        if (_has_bias)
            y.add_(_b);
        y = y.swap(0, 1);
        // y: [batch_size, groups, out_height, out_width, out_channels_per_group]
        y = y.move_forward(4, 1, 2);
        // y: [batch_size, groups, out_channels_per_group, out_height, out_width]
        y = y.merge_dim(1, 2);
        // y: [batch_size, out_channels, out_height, out_width]
        return y;
    }
};