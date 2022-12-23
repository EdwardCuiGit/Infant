#include "../../inc/1.functors/un_func.h"
#include "../../inc/1.functors/tensor_node.h"

void UnFunctor::forward(const TensorD<double> &x, TensorList &y) const
{
    assert(y.size() == 1); //Note: we assume y is empty before forward()
    //y.reserve(1); // and now let's setup y's TensorD
    this->forward(x, y[0].data());
}

void UnFunctor::backward(const TensorD<double> &x, const TensorList &y, TensorD<double> &x_grad) const
{
    assert(y.size() == 1); // Note: we assume both x & y are size() == 1 here
    this->backward(x, y[0].data(), y[0].grad(), x_grad);
}
