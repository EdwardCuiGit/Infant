#include "../../inc/1.functors/functor.h"
#include "../../inc/1.functors/tensor_node.h"

void Combine::forward(const TensorList &x, TensorList &y) const
{
    y.reserve(1);
    y[0] = Tensor::Weak_Upgrade(TensorD<double>::combine(Tensor::Weak_Data_Downgrade(x), _first_dims));
}

void Combine::backward(TensorList &x, const TensorList &y)
{
    assert(y.size() == 1);
    TensorDArray<double> x_data, x_grad;
    Tensor::Weak_Both_Downgrade(x, x_data, x_grad);
    TensorD<double>::Combine_Grad(x_data, y[0].data(), y[0].grad(), x_grad, _first_dims);
    // note: no need to upgrad back to Tensor for x_grad as they are all shared data, including dims
}