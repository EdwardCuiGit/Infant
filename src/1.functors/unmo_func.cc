#include "../../inc/1.functors/unmo_func.h"
#include "../../inc/1.functors/tensor_node.h"

void UnMoFunctor::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() == 1);

    forward(x[0].data(), y);
}

void UnMoFunctor::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() == 1);

    backward(x[0].data(), y, x[0].grad());
}

void Divide::forward(const TensorD<double> &x, TensorList &y) const
{
    TensorDArray<double> y1;
    x.divide(y1, _first_match_dims);
    y = Tensor::Weak_Upgrade(y1); //catious: changed y addr here
}

void Divide::backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const
{
    TensorDArray<double> y1, y1_grad;
    Tensor::Weak_Both_Downgrade(y, y1, y1_grad);

    x.divide_grad(y1, y1_grad, x_grad, _first_match_dims);
}

/*void MeanVar::forward(const TensorD<double> &x, TensorList &y) const
{
    y.reserve(2);
    x.mean_var(y[0].data(), y[1].data(), _biased, _last_work_dims);
}

void MeanVar::backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const
{
    assert(y.size() == 2);
    x.mean_var_grad(y[0].data(), y[0].grad(), y[1].data(), y[1].grad(), x_grad, _biased, _last_work_dims);
}*/