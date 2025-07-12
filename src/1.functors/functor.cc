#include "inc/1.functors/functor.h"
#include "inc/1.functors/tensor_node.h"

int Functor::check_isnan(const TensorList &x) const
{
    for (uint i = 0; i < x.size(); ++i)
    {
        if (x[i].data().has_nan())
        {
            return i;
        }
    }

    return -1;
}

void Combine::forward(const TensorList &x, TensorList &y) const
{
    y.reserve(1);
    y[0] = Tensor::Weak_Upgrade(TensorD<float>::combine(Tensor::Weak_Data_Downgrade(x), _first_dims));
    assert(check_isnan(y) == -1);
}

void Combine::backward(TensorList &x, const TensorList &y)
{
    assert(y.size() == 1);
    TensorDArray<float> x_data, x_grad;
    Tensor::Weak_Both_Downgrade(x, x_data, x_grad);
    TensorD<float>::Combine_Grad(x_data, y[0].data(), y[0].grad(), x_grad, _first_dims);
    // note: no need to upgrad back to Tensor for x_grad as they are all shared data, including dims
    assert(check_isnan(x) == -1);
}

void Where::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() == 1);
    auto res = x[0].data().where(_type, _v);

    y.reserve(2);
    for (uint i = 0; i < 2; i++)
    {
        y[i] = Tensor::Weak_Upgrade(res[i]);
    }
    assert(check_isnan(y) == -1);
}

// no grad for indices, so this backward will not generate grad for x[0]
void Where::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() == 1);
    assert(check_isnan(x) == -1);
}

void TopK::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() == 1);
    auto res = x[0].data().topk(_k);
    assert(res.size() == 2);
    y.reserve(2);
    for (uint i = 0; i < 2; i++)
    {
        y[i] = Tensor::Weak_Upgrade(res[i]);
    }
    assert(check_isnan(y) == -1);
}

void TopK::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() == 1);
    assert(y.size() == 2);
    x[0].data().topk_grad(y[0].data(), y[1].grad(), x[0].grad(), _k);
    assert(check_isnan(x) == -1);
}

void Index::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() >= 2);
    auto res = x[0].data().index(Tensor::Weak_Data_Downgrade(x.subset(1)), _cross);
    y.reserve(1);
    y[0] = Tensor::Weak_Upgrade(res);
    assert(check_isnan(y) == -1);
}

//  note: no grad for indices, there are grads for x[0]
void Index::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() >= 2);
    x[0].data().index_grad(Tensor::Weak_Data_Downgrade(x.subset(1)), y[0].grad(), x[0].grad(), _cross);
    assert(check_isnan(x) == -1);
}

void Assign::forward(const TensorList &x, TensorList &y) const
{
    assert(x.size() == 3);
    auto res = x[0].data().assign(x[1].data(), x[2].data());
    y.reserve(1);
    y[0] = Tensor::Weak_Upgrade(res);
    assert(check_isnan(y) == -1);
}

//  note: no grad for indices, there are grads for x[1]
void Assign::backward(TensorList &x, const TensorList &y)
{
    assert(x.size() == 3);
    x[0].data().assign_grad(x[1].data(), x[2].data(), y[0].grad(), x[1].grad());
    assert(check_isnan(x) == -1);
}

