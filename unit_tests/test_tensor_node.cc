#pragma once
#include "unit_test.h"
#include "../inc/1.functors/tensor_node.h"

/*
1) binary map functors:
add: Vector(_), Tensor, Functor
mul: Vector(_), Tensor, Functor
2) binary reduce functors:
dot: Vector, Tensor, Functor
mse: Vector, Tensor, Functor
ce:  Vector, Tensor, Functor
bce: Vector
re:  Vector
cosine_distance: Vector, Tensor, Functor
euclidean: Vector, Tensor, Functor
match_bottom: Vector
3) unary map functors:
linear: Vector(_), Tensor, Functor
sqrt: Tensor, Functor
pow: Tensor, Functor
softmax: Vector, Tensor, Functor
activation: Tensor, Functor
4) unary reduce functors:
sum: Vector, Tensor, Functor
product: Vector
avg: Vector, Tensor, Functor
mean_var: Vector
var: Tensor, Functor
max: Vector, Tensor, Functor
min: Vector, Tensor, Functor
norm_ln: Vector
norm_l1: Vector
entropy: Vector
binary_entropy: Vector
all_pos: Vector
all_in01: Vector
5) manipulation functors
swap: Array, Tensor, Functor
move_forward: Array, Tensor, Functor
//dropout: Functor
img2col: Tensor, Functor 
divide: Tensor, Functor
combine: Tensor, Functor
MergeDim: Tensor, Functor
Inflate: Tensor, Functor
Squeeze: Tensor, Functor
*/
class TestTensorNode : public TestClass
{
public:
    REGISTER_TEST_CASES(test_creates, test_upgrades, test_downgrades, test_gets, test_add, test_dot, test_linear, test_var, test_divide, test_combine)

    static void test_creates()
    {
        Tensor t1, t2({2, 3}, TensorInit_Types::One, "input");
        assert(t1.size() == 0);
        assert(t1.dim().equals_to({}));
        assert(t2.size() == 6);
        assert(t2.data().vector()[1] == 1);
        assert(t2.id() == "input");

        Tensor t3(t2);
        assert(t3.size() == 6);
    }

    static void test_upgrades()
    {
        TensorD<double> td({2, 3}, TensorInit_Types::Ordinal);
        Tensor t = Tensor::deep_upgrade(td);
        assert(t.size() == 6);
        assert(t.data().vector()[1] = 1);
        t.data().vector()[1] = 2;
        assert(td[1] == 1);

        Tensor t1 = Tensor::weak_upgrade(td);
        assert(t1.size() == 6);
        assert(t1.data().vector()[1] = 1);
        t1.data().vector()[2] = 3;
        assert(td[2] == 3);

        TensorDArray<double> x({td});
        TensorList t2 = Tensor::weak_upgrade(x);
        assert(t2.size() == 1);
        assert(t2[0].data().vector()[1] = 1);
    }

    static void test_downgrades()
    {
        Tensor t1({2, 3}, TensorInit_Types::Ordinal);
        Tensor t2({2, 2}, TensorInit_Types::Ordinal);
        TensorDArray<double> data, grad;
        Tensor::weak_both_downgrade({t1, t2}, data, grad);
        assert(data[0].size() == 6);
        assert(data[1].size() == 4);
        assert(grad[0].size() == 6);
        assert(grad[1].size() == 4);
        assert(data[0].vector()[1] == 1);
        assert(grad[0].vector()[1] == 0);

        TensorList x{t1, t2};
        data = Tensor::weak_data_downgrade(x);
        assert(data[0].size() == 6);
        assert(data[1].size() == 4);
        assert(data[0].vector()[1] == 1);
    }

    static void test_gets()
    {
        Tensor t1({2, 3}, TensorInit_Types::Ordinal);
        assert(t1.dim().equals_to({2, 3}));
        assert(t1.shape() == 2);
        assert(t1.size() == 6);
        assert(t1.size_to_dim(2) == 1);
    }

    static void test_add()
    {
        Tensor x1({1, 2, 3}, TensorInit_Types::Ordinal);
        Tensor x2({1, 3, 3}, TensorInit_Types::Ordinal);

        x1.data().vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.data().vector().set(0, {8, 7, 6, 5, 4, 3, 2, 1, 0});

        auto y = x1.add(x2, 1, 1, 0, 1, 1);
        assert(y.dim().equals_to({1, 2, 3, 3}));
        assert(y.data().vector().equals_to({8, 8, 8, 5, 5, 5, 2, 2, 2, 11, 11, 11, 8, 8, 8, 5, 5, 5}));

        auto y1 = x1.add_(x2, 1, 1, 0, 1, 1);
        assert(x1.dim().equals_to({1, 2, 3, 3}));
        assert(x1.data().vector().equals_to({8, 8, 8, 5, 5, 5, 2, 2, 2, 11, 11, 11, 8, 8, 8, 5, 5, 5}));
        assert(y1.dim().equals_to({1, 2, 3, 3}));
        assert(y1.data().vector().equals_to({8, 8, 8, 5, 5, 5, 2, 2, 2, 11, 11, 11, 8, 8, 8, 5, 5, 5}));
    }

    static void test_dot()
    {
        Tensor x1({1, 2, 3}), x2({1, 3, 3}), y;
        x1.data().vector().set(0, {0, 1, 2, 3, 4, 5});
        x2.data().vector().set(0, {8, 7, 6, 5, 4, 3, 2, 1, 0});

        y = x1.dot(x2, 1, 1);
        assert(y.dim().equals_to({1, 2, 3}));
        assert(y.data().vector().equals_to({7 + 12, 4 + 6, 1, 24 + 28 + 30, 15 + 16 + 15, 6 + 4}));
    }

    static void test_linear()
    {
        Tensor x1({1,  4});
        x1.data().vector().set(0, {0, 1, 4, 9});

        Tensor y = x1.linear(2, 1);
        assert(y.data().vector().equals_to({1, 3, 9, 19}));
        x1.linear_(2, 1);
        assert(x1.data().vector().equals_to({1, 3, 9, 19}));
    }

    static void test_var()
    {
        Tensor x({2, 2}, TensorInit_Types::Ordinal), y;
        TensorD<double> y1, y_grad({2}, TensorInit_Types::One), x_grad;
        y = x.var(false, 1);
        assert(y.data().vector().equals_to({0.25, 0.25}));

        Var var(false, 1);
        var.forward(x.data(), y1);
        assert(y1.vector().equals_to({0.25, 0.25}));
        var.backward(x.data(), y1, y_grad, x_grad);
        assert(x_grad.vector().equals_to({-0.25, 0.25, -0.25, 0.25}));
    }

    static void test_divide()
    {
        Tensor x({2, 3, 2}, TensorInit_Types::Ordinal);
        auto y = x.divide();

        assert(y.size() == 2);
        assert(y[0].dim().equals_to({3, 2}));
        assert(y[0].data().vector().equals_to({0, 1, 2, 3, 4, 5}));

        assert(y[1].dim().equals_to({3, 2}));
        assert(y[1].data().vector().equals_to({6, 7, 8, 9, 10, 11}));

        Divide divide;

        y[0].grad().reset({3, 2}, TensorInit_Types::One);
        y[1].grad().reset({3, 2}, TensorInit_Types::One);

        divide.backward(x.data(), y, x.grad());

        assert(x.grad().dim().equals_to({2, 3, 2}));
        assert(x.grad().vector().equals_to(Vector<double>::one(12)));
    }

    static void test_combine()
    {
        TensorList x;
        x.reserve(2);
        x[0].reset({3, 2}, TensorInit_Types::Ordinal);
        x[1].reset({3, 2}, TensorInit_Types::Ordinal);

        Tensor y = Tensor::combine(x);

        assert(y.dim().equals_to({2, 3, 2}));
        assert(y.data().vector().equals_to({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));

        y.grad().reset({2, 3, 2}, TensorInit_Types::One);

        Combine combine;
        combine.backward(x, {y});

        assert(x.size() == 2);
        assert(x[0].grad().dim().equals_to({3, 2}));
        assert(x[1].grad().vector().equals_to(Vector<double>::one(6)));
    }
};