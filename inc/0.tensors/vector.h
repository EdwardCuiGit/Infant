#pragma once

#include <random>
#include "array.h"

// Vector is basic class for numeric array, support major binary & unary map/reduce ops
// alowed T: int, float, short, byte
// note: good coding style, low compute efficency
// out and v1 may have overlap in memory, out needs to be set to correct size
// TODO: test on float/int/short types
// TODO: are all the functions working well when size() == 0

/*
    core ops: map, reduce, bool_func for both binary &

    supported creation ops:
        ctor: default, elem-list, size + init_type, copy ctor
        init, copy, clear
        zero, one, ordinal, set_dist/uniform/gaussian/lognormal/cauthy/chi_squared/fisher_f/student_t;

    // by default, not supporting broadcasting for all vector ops
    /*supported bin ops include:
        map:
            map, add/add_grad, mul/mul_grad,
            Vector<T>& add(const Vector<T>& v2, Vector<T>& out, uint v1_start = 0, uint v2_start = 0, uint out_start = 0, int len = 0, bool add_to = false) const;
        reduce:
            num: reduce, sum_func, dot, mse, ce/bce, relative_entropy, cosine_distance, euclidean
            bool: bool_func, match_bottom
            T dot(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = 0) const;
    supported uni ops include:
        map: map, linear/*linear_, softmax/*softmax_, *activation/activation_, *sqrt, *pow,
            TODO: log, exp, sin/cos, asin/acos, sort, topk, clamp
            bool TODOs: eq, ge/gt, le/lt,
            probability TODOs: histc(histogram), erf;
            TODO: spectral ops, bit ops, matrix ops, BLAS/LAPACK
            Vector<T>& softmax(Vector<T>& out, uint v1_start = 0, uint out_start, int len = -1) const;
        reduce:
            num: reduce, sum_func, sum, product, avg, *var, max, min, norm_ln/norm_l1, entropy/binary_entropy, TODO: median,
            bool: all_pos, all_in01
            double &sum(uint v1_start = 0, int len = -1) const;
    */

/*// random init, called in main function's starting
// TODO: ensure it thread safe
void __global_random_init()
{
    // TODO
}*/

class Random
{
public:
    static std::default_random_engine __global_random_generator;
};

std::default_random_engine Random::__global_random_generator;


template <class T>
class Vector : public Array<T>
{
public:
    /***********************************************************************************************************/
    // below are vector creation functions

    Vector() : Array<T>() {}

    Vector(std::initializer_list<T> l) : Array<T>(l) {}

    explicit Vector(uint size, TensorInit_Types init_type = TensorInit_Types::None) : Array<T>(size)
    {
        init(init_type);
    }

    /*explicit*/ Vector(const Vector<T> &v2)
    {
        this->copy(v2);
    }

    explicit Vector(const Array<T> &v2)
    {
        this->copy(v2);
    }

    inline static Vector<T> Zero(uint size)
    {
        return Vector<T>(size, TensorInit_Types::Zero);
    }

    inline static Vector<T> One(uint size)
    {
        return Vector<T>(size, TensorInit_Types::One);
    }

    inline static Vector<T> Ordinal(uint size)
    {
        return Vector<T>(size, TensorInit_Types::Ordinal);
    }

    inline static Vector<T> Gaussian(uint size)
    {
        return Vector<T>(size, TensorInit_Types::Gaussian);
    }

    inline static Vector<T> Rand(uint size)
    {
        return Vector<T>(size, TensorInit_Types::Rand);
    }

    Vector<T> &init(TensorInit_Types init_type = TensorInit_Types::None)
    {
        switch (init_type)
        {
        case TensorInit_Types::None:
            break;
        case TensorInit_Types::Zero:
            this->set_each(0.0);
            break;
        case TensorInit_Types::One:
            this->set_each(1.0);
            break;
        case TensorInit_Types::Half:
            this->set_each(0.5);
            break;
        case TensorInit_Types::Ordinal:
            this->set_ordinal();
            break;
        case TensorInit_Types::Gaussian:
            this->set_dist_gaussian(0, 1);
            break;
        case TensorInit_Types::Rand:
            this->set_dist_uniform(0, 1);
            break;
        default:
            assert(false);
        }

        return *this;
    }

    void set_ordinal()
    {
        for (uint i = 0; i < this->size(); ++i)
        {
            (*this)[i] = i;
        }
    }

    // below are random distribution based set, supported all C++ supported distributions in STL
    void set_dist(const std::function<T()> &next_func, uint v1_start = 0, int len = -1)
    {
        if (len < 0)
            len = this->size() - v1_start;
        for (uint i = 0; i < len; ++i)
        {
            (*this)[v1_start + i] = next_func();
        }
    }

    // deprecated, no tested
    // uniform distribution in [0 ~ n]
    /*void set_randn(uint n, uint v1_start, uint len)
    {
        std::srand(std::time(nullptr));
        for (uint i = 0; i < len; ++i)
        {
            T e = 1.0 * std::rand() / (RAND_MAX + 1u) * n;
            (*this)[v1_start + i] = e;
        }
    }*/

    // TODO: only supports double now
    void set_dist_uniform(double min, double max, uint v1_start = 0, int len = -1)
    {
        std::uniform_real_distribution<double> dist(min, max);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_gaussian(double mean, double var, uint v1_start = 0, int len = -1)
    {
        std::normal_distribution<double> dist(mean, var);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_lognormal(T m, T s, uint v1_start = 0, int len = -1)
    {
        std::lognormal_distribution<double> dist(m, s);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_cauchy(T n, uint v1_start = 0, int len = -1)
    {
        std::chi_squared_distribution<double> dist(n);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_chi_squared(T n, uint v1_start = 0, int len = -1)
    {
        std::chi_squared_distribution<double> dist(n);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_fisher_f(T m, T n, uint v1_start = 0, int len = -1)
    {
        std::fisher_f_distribution<double> dist(m, n);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    void set_dist_student_t(T n, uint v1_start = 0, int len = -1)
    {
        std::student_t_distribution<double> dist(n);
        set_dist([&dist]() -> T
                 { return dist(Random::__global_random_generator); },
                 v1_start, len);
    }

    /****************************************************************************************************************/
    /*  below are major vector math ops
    add is to output results to out tensor, if add_to is enabled, will keep original value in out; return value is out's ref';
    add_ is to output results to self, may change shape() for some ops, return value is *this's ref, add_to is false in this case;
    above applies to all ops, except explicit annoucements;
    */

    // 1) below are binary map ops: vector op vector => vector

    Vector<T> &map(const std::function<T(T, T)> &map_func, const Vector<T> &v2, Vector<T> &out,
                   uint v1_start = 0, uint v2_start = 0, uint out_start = 0, int len = -1, bool add_to = false) const
    {
        if (len < 0)
            len = v2.size() - v2_start;
        assert(v1_start + len <= this->size());
        assert(v2_start + len <= v2.size());

        for (uint j = 0; j < len; ++j)
        {
            T res = map_func(this->get(v1_start + j), v2[v2_start + j]);
            if (std::isnan(res))
            {
                assert(false);
            }
            if (add_to)
                out[out_start] += res;
            else
                out[out_start] = res;
            ++out_start;
        }

        return out;
    }

    // deprecated, no broadcast support
    /*// support broadcast, for each unit in v2, we will do pointwise ops in v1, for v1_unit times
    // we assume out already allocated enough capacity
    // no problem if &out == this
    // what if unit_len || v1_unit == 0? => do nothing
    Vector<T> &bc_map(const std::function<T(T, T)> &map_func, const Vector<T> &v2, Vector<T> &out,
                    uint v1_start = 0, uint v2_start = 0, uint out_start = 0, int unit_len = -1, uint v1_unit = 1, bool add_to = false) const
    {
        if (unit_len < 0)
            unit_len = v2.size() - v2_start;
        assert(v1_start + v1_unit * unit_len <= size());
        assert(v2_start + unit_len <= v2.size());
        assert(out_start + v1_unit + unit_len < out.size());

        for (uint i = 0; i < v1_unit; ++i)
        {
            map(map_func, v2, out, v1_start, v2_start, out_start, unit_len, add_to);
            out_start += unit_len;
        }

        return out;
    }*/

    // add to out/self, not support broadcasts
    // out[i] = v1[i] * alpha_v1 + v2[i] * alpha_v2 + beta
    // TODO: most case alpha & beta not used, how to make perf better?
    Vector<T> &add(const Vector<T> &v2, Vector<T> &out, uint v1_start = 0, uint v2_start = 0, uint out_start = 0, int len = -1,
                   bool add_to = false, T alpha_v1 = 1, T alpha_v2 = 1, T beta = 0) const
    {
        // TODO: perf: most case alpha, beta is not assigned
        return map([alpha_v1, alpha_v2, beta](T e1, T e2) -> T
                   { return e1 * alpha_v1 + e2 * alpha_v2 + beta; },
                   v2, out, v1_start, v2_start, out_start, len, add_to);
    }

    inline Vector<T> &add_(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1,
                           T alpha_v1 = 1, T alpha_v2 = 1, T beta = 0)
    {
        return add(v2, *this, v1_start, v2_start, v1_start, len, false, alpha_v1, alpha_v2, beta);
    }

    // vector grad funcs are all deprecated
    // v1_grad[i] = out_grad[i] * alpha_v1
    // TODO support full interface
    /*Vector<T> &add_grad(const Vector<T> &out_grad, T alpha_v1, Vector<T> &v1_grad) const
    {
        out_grad.linear(v1_grad, 0, 0, -1, alpha_v1);
        return v1_grad;
    }*/

    // out[i] [+]= v1[i] * v2[i] * alpha + beta;
    Vector<T> &mul(const Vector<T> &v2, Vector<T> &out, uint v1_start = 0, uint v2_start = 0, uint out_start = 0, int len = -1,
                   bool add_to = false, T alpha = 1, T beta = 0) const
    {
        // TODO: perf: most case alpha, beta is not assigned
        return map([alpha, beta](T e1, T e2) -> T
                   { return e1 * e2 * alpha + beta; },
                   v2, out, v1_start, v2_start, out_start, len, add_to);
    }

    inline Vector<T> &mul_(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1,
                           T alpha = 1, T beta = 0)
    {
        return mul(v2, *this, v1_start, v2_start, v1_start, len, false, alpha, beta);
    }

    // v1_grad[i] = out_grad[i] * v2[i] * alpha
    // TODO: support full interface
    /*Vector<T> &mul_grad(const Vector<T> &out_grad, const Vector<T> &v2, T alpha, Vector<T> &v1_grad) const
    {
        out_grad.mul(v2, v1_grad, 0, 0, 0, -1, false, alpha);
        return v1_grad;
    }*/

    // 2) binary reduce ops: vector op vector => float

    T reduce(const std::function<T(T, T, uint)> &map_func, const std::function<T(T, T, uint)> &reduce_func,
             const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        if (len < 0)
            len = v2.size() - v2_start;
        assert(len > 0);
        assert(v1_start + len <= this->size());
        assert(v2_start + len <= v2.size());

        T res = map_func(this->get(v1_start), v2[v2_start], 0);
        for (uint i = 1; i < len; ++i)
        {
            T map_res = map_func(this->get(v1_start + i), v2[v2_start + i], i);
            if (std::isnan(map_res))
            {
                assert(false);
            }
            
            res = reduce_func(res, map_res, i);
        }

        return res;
    }

    SUGAR T reduce(const std::function<T(T, T)> &map_func, const std::function<T(T, T, uint)> &reduce_func,
                   const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        return reduce([map_func](T x, T y, uint) -> T
                      { return map_func(x, y); },
                      reduce_func, v2, v1_start, v2_start, len);
    }

    // (vector, vector) -> float
    SUGAR inline T sum_func(const std::function<T(T, T, uint)> &map_func, const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        return reduce(map_func, Math<T>::sum_reduce, v2, v1_start, v2_start, len);
    }

    SUGAR inline T sum_func(const std::function<T(T, T)> &map_func, const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        return reduce([map_func](T x1, T x2, uint) -> T
                      { return map_func(x1, x2); },
                      Math<T>::sum_reduce, v2, v1_start, v2_start, len);
    }

    // (vector, vector) -> float
    T dot(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        return sum_func(Math<T>::multi_op, v2, v1_start, v2_start, len);
    }

    double mse(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        if (len < 0)
            len = v2.size() - v2_start;
        auto res = sum_func([](T x, T y) -> T
                            { return (x - y) * (x - y); },
                            v2, v1_start, v2_start, len);
        return len > 0 ? res / len : 0.0;
    }

    // cross entropy, q(v2) values need to be positive
    double ce(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        assert(v2.all_pos(v2_start, len));
        return -1.0 * sum_func([](T px, T qx) -> T
                               { return px * std::log2(qx); },
                               v2, v1_start, v2_start, len);
    }

    // binary cross entropy, this/y needs to be binary distribution, py(v2) values need to be positive, loss function
    double bce(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        if (len < 0)
            len = v2.size() - v2_start;
        assert(v2.all_in01(false, v2_start, len));
        auto res = -1.0 * sum_func([](T yi, T pyi) -> T
                                   { return yi * std::log2(pyi) + (1 - yi) * std::log2(1 - pyi); },
                                   v2, v1_start, v2_start, len);
        return len > 0 ? res / len : 0;
    }

    // KL divergence/Relative Entropy, p is truth, q is prediction
    double re(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        assert(all_pos(v1_start, len));
        assert(v2.all_pos(v2_start, len));
        return sum_func([](T px, T qx) -> T
                        { return px * std::log2(px / qx); },
                        v2, v1_start, v2_start, len);
    }

    // note: 3 loops here, one loop could be faster;
    double cosine_distance(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        auto norm_v1 = norm_ln(2.0, v1_start, len);
        auto norm_v2 = v2.norm_ln(2.0, v2_start, len);
        norm_v1 = std::max(EPSILON, norm_v1);
        norm_v2 = std::max(EPSILON, norm_v2);
        return dot(v2, v1_start, v2_start, len) / (norm_v1 * norm_v2);
    }

    double euclidean(const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        auto res = sum_func([](T x, T y) -> T
                            { return (x - y) * (x - y); },
                            v2, v1_start, v2_start, len);
        return std::sqrt(res * 1.0);
    }

    // 3) binary bool ops: vector op vector => bool

    // (vector, vector) -> bool
    bool bool_func(const std::function<bool(T, T, uint)> &func, const Vector<T> &v2, uint v1_start = 0, uint v2_start = 0, int len = -1) const
    {
        if (len < 0)
            len = v2.size() - v2_start;
        assert(v1_start + len <= this->size());
        assert(v2_start + len <= v2.size());

        for (uint i = 0; i < len; ++i)
        {
            if (!func(this->get(v1_start + i), v2[v2_start + i], i))
                return false;
        }

        return true;
    }

    // last n elems are the same
    bool match_bottom(const Vector<T> &v2, int last_n = -1, bool bottom = true) const
    {
        // assert(this->size() >= v2.size()); // TODO: we need to consider last_n
        if (this->size() >= v2.size())
        {
            last_n = (last_n == -1 || last_n > v2.size()) ? v2.size() : last_n;
            return bool_func([](T e1, T e2, uint) -> bool
                            { return e1 == e2; },
                            v2, bottom ? this->size() - last_n : 0, bottom ? v2.size() - last_n : 0, last_n);
        }
        else
        {
            last_n = (last_n == -1 || last_n > this->size()) ? this->size() : last_n;
            return bool_func([](T e1, T e2, uint) -> bool
                            { return e1 == e2; },
                            v2, bottom ? this->size() - last_n : 0, bottom ? v2.size() - last_n : 0, last_n);
        }
    }

    // 4. unary map ops: (vector) -> vector
    // TODO: support add_to
    Vector<T> &map(const std::function<T(T)> &func, Vector<T> &out, uint v1_start = 0, uint out_start = 0, int len = -1,
                   bool add_to = false) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        assert(v1_start + len <= this->size());

        for (uint j = 0; j < len; ++j)
        {
            T res = func(this->get(v1_start + j));
            if (std::isnan(res))
            {
                assert(false);
            }
            if (add_to)
                out[out_start] += res;
            else
                out[out_start] = res;
            ++out_start;
        }

        return out;
    }

    // special case of add, linear function, could be used for add_scalar as well
    Vector<T> &linear(Vector<T> &out, uint v1_start = 0, uint out_start = 0, int len = -1, T alpha_v1 = 1, T beta = 0) const
    {
        // TODO: perf: most case alpha, beta is not assigned
        return map([alpha_v1, beta](T e1) -> T
                   { return e1 * alpha_v1 + beta; },
                   out, v1_start, out_start, len, false);
    }

    Vector<T> &linear_(uint v1_start = 0, int len = -1, T alpha_v1 = 1, T beta = 0)
    {
        return linear(*this, v1_start, v1_start, len, alpha_v1, beta);
    }

    Vector<T> &softmax(Vector<T> &out, uint v1_start = 0, uint out_start = 0, int len = -1) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        
        // Find max value first to prevent overflow
        T max_val = this->max(v1_start, len);
        if (max_val == INF_NEG)
        {
            for (uint i = 0; i < len; ++i)            
            {
                out[out_start + i] = 0;
            }

            return out;
        }
        
        double denominator = 0;
        map([max_val, &denominator](T e) -> T
            {
                auto exp_e = std::exp(e - max_val); // Subtract max_val before exp
                denominator += exp_e;
                return exp_e;
            },
            out, v1_start, out_start, len, false);

        return out.map([denominator](T e) -> T
                       { return e / denominator; },
                       out, out_start, out_start, len, false);
    }

    Vector<T> &softmax_(uint v1_start = 0, int len = -1)
    {
        return softmax(*this, v1_start, v1_start, len);
    }

    // 5. unary reduce op: (vector> -> float
    T reduce(const std::function<T(T)> &map_func, const std::function<T(T, T, uint)> &reduce_func,
             uint v1_start = 0, int len = -1) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        assert(len > 0);
        assert(v1_start + len <= this->size());

        T res = map_func(this->get(v1_start));
        for (uint i = 1; i < len; ++i)
        {
            T map_res = map_func(this->get(v1_start + i));
            if (std::isnan(map_res))
            {
                assert(false);
            }
            res = reduce_func(res, map_res, i);
        }

        return res;
    }

    SUGAR inline T reduce(const std::function<T(T)> &map_func, const std::function<T(T, T)> &reduce_func,
                          uint v1_start = 0, int len = -1) const
    {
        return reduce(
            map_func, [reduce_func](T res, T curr, uint) -> T
            { return reduce_func(res, curr); },
            v1_start, len);
    }

    inline T sum_func(const std::function<T(T)> &func, uint v1_start = 0, int len = -1) const
    {
        if (this->size() == 0)
            return 0;
        return reduce(func, Math<T>::sum_reduce, v1_start, len);
    }

    T sum(uint v1_start = 0, int len = -1) const
    {
        return sum_func(Math<T>::empty_map, v1_start, len);
    }

    T product(uint v1_start = 0, int len = -1) const
    {
        if (this->size() == 0)
            return 0;
        return reduce(Math<T>::empty_map, Math<T>::product_reduce, v1_start, len);
    }

    T avg(uint v1_start = 0, int len = -1, T alpha = 1.0, T beta = 0.0) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        auto res = sum_func([alpha, beta](T e) -> T
                            { return alpha * e + beta; },
                            v1_start, len);
        return len > 0 ? res / len : 0;
    }

    NOTEST T var(uint v1_start = 0, int len = -1, T alpha = 1.0, T beta = 0.0) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        
        T mean = avg(v1_start, len, alpha, beta);
        auto res = sum_func([mean](T e) -> T{return (e - mean) * (e - mean);}, v1_start, len);
        return len > 0 ? res / len : 0;
    }

    T max(uint v1_start = 0, int len = -1) const
    {
        return reduce(
            Math<T>::empty_map, [](T res, T e, uint i) -> T &
            { return e > res ? e : res; },
            v1_start, len);
    }

    T min(uint v1_start = 0, int len = -1) const
    {
        return reduce(
            Math<T>::empty_map, [](T res, T e, uint i) -> T &
            { return e < res ? e : res; },
            v1_start, len);
    }

    double norm_ln(double n, uint v1_start = 0, int len = -1) const
    {
        assert(n != 0);
        T res = sum_func([n](T e) -> T
                         { return std::pow(e, n); },
                         v1_start, len);
        return std::pow(res, 1.0 / n);
    }

    double norm_l1(uint v1_start = 0, int len = -1) const
    {
        return sum_func([](T e) -> T
                        { return e >= 0 ? e : -1 * e; },
                        v1_start, len);
    }

    double entropy(uint v1_start = 0, int len = -1) const
    {
        assert(all_pos(v1_start, len));
        return -1.0 * sum_func([](T px) -> T
                               { return px * std::log2(px); },
                               v1_start, len);
    }

    // binary distribution
    static double Binary_Entropy(double p)
    {
        assert(p > 0 && p < 1);
        return -1.0 * (p * std::log2(p) + (1 - p) * std::log2(1 - p));
    }

    // 6. unary bool op: (vector)->bool, not implemented based on _reduce() now
    bool bool_func(const std::function<bool(T)> &map_func, uint v1_start = 0, int len = -1) const
    {
        if (len < 0)
            len = this->size() - v1_start;
        assert(len > 0);
        assert(v1_start + len <= this->size());

        for (uint i = 0; i < len; ++i)
        {
            if (!map_func(this->get(v1_start + i)))
                return false;
        }

        return true;
    }

    // (vector) -> bool
    bool all_pos(uint v1_start = 0, int len = -1) const
    {
        return bool_func([](T e) -> bool
                         { return e > 0; },
                         v1_start, len);
    }

    bool all_in01(bool strict = false, uint v1_start = 0, int len = -1) const
    {
        return bool_func([strict](T e) -> bool
                         { return strict ? e > 0 && e < 1 : e >= 0 && e <= 1; },
                         v1_start, len);
    }

    // manipulation funcs
    static void Flatten(const Vector<Vector<T>>& bins_vector, Vector<T>& flatten_vector)
    {
        for (uint i = 0; i < bins_vector.size(); ++i)
        {
            flatten_vector.append(bins_vector[i]);
        }
    }
};