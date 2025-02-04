#pragma once
#include <cmath> //std::pow, exp, sqrt, max, min, log2

// below are single elem math ops, all static funcs
// no unit test as all tested in test_vector.cc or test_tensor.cc
template <class T>
class Math
{
public:
    inline static bool almost_equal(T e1, T e2, T noise_level = 0.00001)
    {
        if (e1 == e2)
            return true;
        T diff = e1 - e2;
        if (diff <= noise_level && diff >= -noise_level)
            return true;
        return false;
    }

    inline static void assert_almost_equal(T e1, T e2, T noise_level = 0.00001)
    {
        assert(almost_equal(e1, e2, noise_level));
    }

    inline static bool almost_zero(T e, T noise_level = 0.00001)
    {
        if (noise_level < 0) noise_level = -noise_level;

        return e <= noise_level && e >= -noise_level;
    }

    inline static void swap(T &e1, T &e2)
    {
        T tmp;
        tmp = e2;
        e2 = e1;
        e1 = tmp;
    }

    inline static T empty_map(T e)
    {
        return e;
    }

    inline static T relu(T e)
    {
        return e > 0 ? e : 0;
    }

    inline static T relu_grad(T x, T y)
    {
        return x > 0 ? 1 : 0;
    }

    inline static T sigmoid(T x)
    {
        return 1.0 / (1.0 + std::exp(-1.0 * x));
    }

    inline static T sigmoid_grad(T x, T y)
    {
        // return y * y * std::exp(-1.0 * x);
        return y * (1 - y);
    }

    inline static T tanh(T x)
    {
        // return 2 * sigmoid(2 * x) - 1;
        return 2.0 / (1.0 + std::exp(-2.0 * x)) - 1;
    }

    inline static T tanh_grad(T x, T y)
    {
        // return 2 * -1 * ((y+1)/2) ^ 2 * std::exp(-2.0 * x) * -2;
        // return (y+1) * (y+1) * std::exp(-2.0 * x);
        return 1 - y * y;
    }

    inline static T sqrt(T e)
    {
        return std::sqrt(e);
    }

    inline static T add_op(T e1, T e2)
    {
        return e1 + e2;
    }

    inline static T multi_op(T e1, T e2)
    {
        return e1 * e2;
    }

    inline static T empty_reduce(T res, T e, uint)
    {
        return res;
    }

    inline static T sum_reduce(T pre_res, T e, uint)
    {
        return pre_res + e;
    }

    inline static T sum_reduce0(T pre_res, T e)
    {
        return pre_res + e;
    }

    inline static T product_reduce(T pre_res, T e, uint)
    {
        return pre_res * e;
    }
};