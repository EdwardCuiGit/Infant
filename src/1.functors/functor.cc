#include "../../inc/1.functors/functor.h"
#include "../../inc/1.functors/tensor_node.h"

void Combine::forward(const TensorList &x, TensorList &y) const
{
    y.reserve(1);
    y[0] = Tensor::weak_upgrade(TensorD<double>::combine(Tensor::weak_data_downgrade(x), _first_dims));
}

void Combine::backward(TensorList &x, const TensorList &y)
{
    assert(y.size() == 1);
    TensorDArray<double> x_data, x_grad;
    Tensor::weak_both_downgrade(x, x_data, x_grad);
    TensorD<double>::combine_grad(x_data, y[0].data(), y[0].grad(), x_grad, _first_dims);
    // note: no need to upgrad back to Tensor for x_grad as they are all shared data, including dims
}