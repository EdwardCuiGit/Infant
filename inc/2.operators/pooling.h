#pragma once

#include "operator_base.h"

/*
aggregation op: vector[N, I] -> float[N]
1) max, min, avg
// TODO: Pooling1d
*/
class Pooling2d : public UnOp
{
private:
    Pooling_Types _pt;
    uint _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y;

public:
    Pooling2d(Pooling_Types pt, uint kernel_x, uint kernel_y, uint stride_x = 1, uint stride_y = 1, uint padding_x = 0, uint padding_y = 0)
        : _pt(pt), _kernel_x(kernel_x), _kernel_y(kernel_y), _stride_x(stride_x), _stride_y(stride_y),
          _padding_x(padding_x), _padding_y(padding_y)
    {
    }

    // x: [N: batch_size, C, H, W] => y: [batch_size, C, H, W]
    virtual void forward(const Tensor &x, Tensor &y) const override
    {
        assert(x.shape() == 4);
        uint batch_size = x.dim()[0], in_channels = x.dim()[1], in_height = x.dim()[2], in_width = x.dim()[3];

        // col: {1, batch_size, out_height, out_width, in_channels, kernel_y, kernel_x}
        Tensor col = x.im2col(1, _kernel_x, _kernel_y, _stride_x, _stride_y, _padding_x, _padding_y);
        Tensor y1;// y1: {1, batch_size, out_height, out_width, in_channels}
        switch (_pt)
        {
        case Pooling_Types::Avg:
            y1 = col.avg(2);
            break;
        case Pooling_Types::Max: 
            y1 = col.max(2);
            break;
        case Pooling_Types::Min:
            y1 = col.min(2);
            break;
        default:
            assert(false);
            break;
        }

        Tensor y2 = y1.move_forward(4, 1, 2); // y: {1, batch_size, in_channels, out_height, out_width};
        y = y2.squeeze();
    }
};