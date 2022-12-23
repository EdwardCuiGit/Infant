#include "../../inc/1.functors/bin_func.h"
#include "../../inc/1.functors/tensor_node.h"

void BinFunctor::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() == 2);
    assert(y.size() == 1);

    forward(x[0].data(), x[1].data(), y[0].data());
}

void BinFunctor::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() == 2);
    assert(y.size() == 1);

    backward(x[0].data(), x[1].data(), y[0].data(), y[0].grad(), x[0].grad(), x[1].grad());
}