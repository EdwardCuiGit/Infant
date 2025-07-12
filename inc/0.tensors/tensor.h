#pragma once

#include "vector.h"
#include "string_util.h"
#include "cuda_ctx.h"

#pragma region comments

#pragma endregion

template <class T>
class TensorD;
typedef Ptr<TensorD<float>> TensorDP;
typedef Array<TensorDP> TensorDPArray;
template <typename T>
using TensorDArray = Array<TensorD<T>>;

template <class T>
class TensorD
{
#pragma region friends
    friend class TestTensor;
    friend class TestTensorNode;
    friend class TestFc;
    friend class TestConv;
    friend class TestPooling;
    friend class TestAttention;
    friend class TestRnn;
    friend class TestNorm;
    friend class TestFunctorGraph;
    friend class TestOptimizers;
    friend class TestDataLoaders;
    friend class Gbdt;
#pragma endregion
private:
    mutable Ptr<Vector<T>> _vector;
    Vector<uint> _dim;
    mutable float *_device_vector = nullptr;
#pragma region creates
public:
    NEEDS_CUDA TensorD()
    {
        this->_vector = std::make_shared<Vector<T>>();
    }

    NEEDS_CUDA TensorD(const TensorD<T> &x2) : TensorD()
    {
        this->weak_copy(x2);
    }

    NEEDS_CUDA explicit TensorD(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None) : TensorD()
    {
        this->reset(dim, t);
    }

    NEEDS_CUDA explicit TensorD(const Vector<uint> &dim, const Vector<T> &vector) : TensorD()
    {
        this->reset(dim, vector);
    }

    NEEDS_CUDA TensorD<T> &reset(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None)
    {
        if (this->size() != dim.product()) // if total size is the same, no need to re-alloc memory, but the data is random
        {
            this->vector_ptr() = std::make_shared<Vector<T>>();
            this->vector().reserve(dim.product());
        }

        if (!this->_dim.equals_to(dim)) // over-optimize, equals_to also spend time
        {
            this->_dim.copy(dim);
        }

        if (t == TensorInit_Types::LEFT_LOWER_ONE)
        {
            /*
            1,0,0
            1,1,0
            1,1,1
            */
            assert(dim.size() == 2);
            assert(dim[0] == dim[1]);
            for (uint i = 0; i < dim[0]; ++i)
                for (uint j = 0; j < dim[1]; ++j)
                {
                    vector()[i * dim[0] + j] = (j <= i ? 1 : 0);
                }
        }
        else if (t == TensorInit_Types::RIGHT_HIGHER_NEG_INF)
        {
            /*
            0, -inf, -inf,
            0, 0,    -inf,
            0, 0,       0
            */
            assert(dim.size() == 2);
            assert(dim[0] == dim[1]);
            for (uint i = 0; i < dim[0]; ++i)
                for (uint j = 0; j < dim[1]; ++j)
                {
                    vector()[i * dim[0] + j] = (j <= i ? 0 : INF_NEG);
                }
        }
        else
        {
            vector().init(t);
        }

        return *this;
    }

    NEEDS_CUDA TensorD<T> &reset(const Vector<uint> &dim, const Vector<T> &vector)
    {
        this->reset(dim);

        assert(this->size() == vector.size());
        this->vector().set(0, vector);
        return *this;
    }

    // this is to deep copy x2 by n times to *this
    // perf: if copy from same shape tensor, no need to clear() and init();
    NEEDS_CUDA TensorD<T> &deep_copy(const TensorD<T> &x2, uint n = 1)
    {
        if (n == 1)
        {
            if (&x2 == this || x2.vector_ptr() == this->vector_ptr())
            {
                return *this;
            }

            this->reset(x2.dim());
            this->vector().set(0, x2.vector());
        }
        else
        {
            if (&x2 == this || x2.vector_ptr() == this->vector_ptr()) // special logic for copy itself
            {
                auto new_dim = Vector<uint>(this->dim().clone().insert(0, n));
                Vector<T> clone;
                clone.copy(this->vector());
                this->reset(new_dim);
                for (uint i = 0; i < n; ++i)
                {
                    this->vector().set(this->size() / n * i, clone, 0, this->size() / n);
                }

                return *this;
            }

            auto new_dim = Vector<uint>(this->dim().clone().insert(0, n));
            this->reset(new_dim);
            for (uint i = 0; i < n; ++i)
            {
                this->vector().set(this->size() / n * i, x2.vector(), 0, this->size() / n);
            }
        }

        return *this;
    }

    NEEDS_CUDA TensorD<T> &weak_copy(const TensorD<T> &x2)
    {
        this->_dim = x2.dim();
        this->vector_ptr() = x2.vector_ptr();
        return *this;
    }

    // this function is pair function of save, pls change together
    NEEDS_CUDA inline void load(std::istream &i)
    {
        this->clear();

        StringUtil::assert_next_line(i, "start of tensor data");
        uint shape_value = StringUtil::read_uint(i, "shape");
        assert(shape_value > 0);

        Vector<uint> dim;
        StringUtil::read_uint_vector(i, dim);

        this->reset(dim);

        assert(this->shape() == shape_value);

        StringUtil::read_float_vector(i, this->vector());

        StringUtil::assert_next_line(i, "end of tensor data");
    }

    NEEDS_CUDA inline void save(std::ostream &o) const
    {
        o << "start of tensor data\n";
        StringUtil::write_uint(o, "shape", this->shape());
        StringUtil::write_vector(o, this->dim());

        StringUtil::write_vector(o, this->vector());

        o << "end of tensor data\n";
    }

    OVERRIDE void clear()
    {
        // this->_vector->clear(); after all reference to _vector is removed, dctor will be called automatically
        // this->_vector = nullptr;
        if (this->_vector != nullptr)
        {
            this->_vector->clear();
        }

        if (this->_device_vector != nullptr)
        {
            CudaCtx<T>::free_device(this->_device_vector);
            this->_device_vector = nullptr;
            this->_vector = std::make_shared<Vector<T>>(); //  needs to ensure one of the vectors is not null
        }

        // this->_dim = nullptr;
        this->_dim.clear();
    }

#pragma endregion
#pragma region retrievals
    /***************************************************************************************************************/
    // below are tensor retrieval functions, all const
    inline const Vector<uint> &dim() const
    {
        return _dim;
    }

    inline Vector<uint> &dim()
    {
        return _dim;
    }

    inline uint size() const
    {
        return dim().product();
    }

    inline uint shape() const
    {
        uint res = dim().size();
        // if (res == 1 && dim()[0] == 1)
        //     res = 0;
        return res;
    }

    NEEDS_CUDA inline T first_item() const
    {
        assert(size() > 0);
        return this->vector()[0];
    }

    NEEDS_CUDA inline bool has_nan() const
    {
        if (_vector == nullptr)
        {
            return false;
        }

        for (uint i = 0; i < size(); ++i)
        {
            if (std::isnan(this->vector()[i]))
            {
                return true;
            }
        }

        return false;
    }

    uint size_to_dim(uint size, bool forward = true) const
    {
        assert(size > 0 && this->size() % size == 0);

        uint dim = 0;
        while (size > 1)
        {
            uint id = dim;
            if (!forward)
                id = shape() - dim - 1;
            size /= this->dim()[id];
            dim++;
        }

        return dim;
    }

    // TODO: use vector.product
    uint dim_to_size(uint from = 0, int len = -1, bool forward = true) const
    {
        assert(from <= shape());
        if (len < 0)
            len = shape() - from;
        else
            assert(from + len <= shape());
        if (!forward)
            from = shape() - from - len;

        uint size = 1;
        for (uint i = 0; i < len; ++i)
        {
            size *= dim()[i + from];
        }



        //
        return size;
    }

    NEEDS_CUDA bool equals_to(const TensorD<T> &x, T noise_level = 0.00001) const
    {
        if (!this->dim().equals_to(x.dim()))
            return false;

        return this->vector().equals_to(x.vector(), noise_level);
    }

    NEEDS_CUDA bool equals_to(const Vector<uint> &dim, const Vector<T> &vector, T noise_level = 0.00001) const
    {
        if (!this->dim().equals_to(dim))
        {
            return false;
        }

        return this->vector().equals_to(vector, noise_level);
    }

#pragma endregion
#pragma region binary_maps

public:
    // add op: y[i] [+]= alpha_x1 * x1[i] + alpha_x2 * x2[i] + beta;
    TensorD<T> &add(const TensorD<T> &x2, TensorD<T> &y, T alpha_x1 = 1.0, T alpha_x2 = 1.0, T beta = 0.0,
                    uint first_match_dims = 0, int last_work_dims = -1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, [alpha_x1, alpha_x2, beta](const float *x1, const float *x2, float *y, uint depths, uint rows_x1, uint rows_x2, uint cols)
                      { CudaCtx<T>::add(x1, x2, y, depths, rows_x1, rows_x2, cols, alpha_x1, alpha_x2, beta); }))
        {
            return y;
        }

        return this->_map([alpha_x1, alpha_x2, beta](T xe1, T xe2) -> T
                          { return xe1 * alpha_x1 + xe2 * alpha_x2 + beta; },
                          x2, y, first_match_dims, last_work_dims);
    }

    TensorD<T> &add_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         T alpha_x1 = 1.0, T alpha_x2 = 1.0, T beta = 0.0,
                         uint first_match_dims = 0, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, [alpha_x1, alpha_x2, beta](const float *x1, const float *x2, const float *y_grad, float *x1_grad, float *x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
                           { CudaCtx<T>::add_grad(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols, alpha_x1, alpha_x2, beta); }, false))
        {
            return x1_grad;
        }

        // x1_grad all values is alpha_x1, x2_grad all values is alpha_x2
        // TODO: most case alpha_x1 & alpha_x2 are 1, perf optimization
        return this->_map_grad([alpha_x1, alpha_x2](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                               {
                            xe1_grad = alpha_x1;
                            xe2_grad = alpha_x2; },
                               x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // mul op: y[i] = alpha * x1[i] * x2[i] + beta; Hamard product
    TensorD<T> &mul(const TensorD<T> &x2, TensorD<T> &y, T alpha = 1.0, T beta = 0.0,
                    uint first_match_dims = 0, int last_work_dims = -1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, [alpha, beta](const float *x1, const float *x2, float *y, uint depths, uint rows_x1, uint rows_x2, uint cols)
                      { CudaCtx<T>::mul(x1, x2, y, depths, rows_x1, rows_x2, cols, alpha, beta); }))
        {
            return y;
        }

        return this->_map([alpha, beta](T xe1, T xe2) -> T
                          { return xe1 * alpha * xe2 + beta; },
                          x2, y, first_match_dims, last_work_dims);
    }

    TensorD<T> &mul_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         T alpha = 1.0, T beta = 0.0,
                         uint first_match_dims = 0, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, [alpha, beta](const float *x1, const float *x2, const float *y_grad, float *x1_grad, float *x2_grad, uint depths, uint rows_x1, uint rows_x2, uint cols)
                           { CudaCtx<T>::mul_grad(x1, x2, y_grad, x1_grad, x2_grad, depths, rows_x1, rows_x2, cols, alpha, beta); }, false))
        {
            return x1_grad;
        }

        // x1_grad all values is alpha * x2[i], x2_grad all values is alpha * x1[i]
        return this->_map_grad([alpha](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                               {
                            xe1_grad = alpha * xe2;
                            xe2_grad = alpha * xe1; },
                               x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }
#pragma endregion

#pragma region binary_reduces
public:
    // dot op: y[i] = sum(j, x1[j] * x2[j])
    TensorD<T> &dot(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, CudaCtx<T>::dot, true))
        {
            return y;
        }

        return _sum_func(Math<T>::multi_op, x2, y, first_match_dims, last_work_dims);
    }

    // x1_grad all values is x2[i], x2_grad all values is x1[i]
    TensorD<T> &dot_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, CudaCtx<T>::dot_grad, true))
        {
            return x1_grad;
        }

        return this->_sum_func_grad([](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                                    {
                                        xe1_grad = xe2;
                                        xe2_grad = xe1; },
                                    x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // mse op: y[i] = avg(j, (x1[j] - x2[j]) ^ 2)
    TensorD<T> &mse(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, CudaCtx<T>::mse, true))
        {
            return y;
        }

        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return _sum_func([n](T xe1, T xe2) -> T
                         { return (xe1 - xe2) * (xe1 - xe2) / n; },
                         x2, y, first_match_dims, last_work_dims);
    }

    // x1_grad all values is 2(x1[i] - x2[i])/n, x2_grad all values is 2(x2[i] - x1[i])/n
    TensorD<T> &mse_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, CudaCtx<T>::mse_grad, true))
        {
            return x1_grad;
        }

        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return this->_sum_func_grad([n](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                                    {
                                 xe1_grad = 2 * (xe1 - xe2) / n;
                                 xe2_grad = -1 * xe1_grad; },
                                    x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // ce op: y[i] = -1.0 * sum(j, (x1[j] * log2(x2[j])))
    TensorD<T> &ce(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, CudaCtx<T>::ce, true))
        {
            return y;
        }

        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return _sum_func([n](T xe1, T xe2) -> T
                         {
            assert(xe2 >= 0);
            xe2 = xe2 == 0 ? xe2 + EPSILON : xe2;
            return -1.0 * xe1 * std::log2(xe2); },
                         x2, y, first_match_dims, last_work_dims);
    }

    // x1_grad all values is log2(x2[j]), x2_grad all values is x1[i] /(x2[i]) / std::log(2)
    TensorD<T> &ce_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                        TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                        uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, CudaCtx<T>::ce_grad, true))
        {
            return x1_grad;
        }

        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return this->_sum_func_grad([n](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                                    {
                        assert(xe2 > 0);
                        xe1_grad = -1.0 * std::log2(xe2);
                        xe2_grad = -1.0 * xe1 / xe2 / std::log(2.0); },
                                    x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // eu op: y[i] = sqrt(sum(j, ((x1[j] - x2[j])^2))
    TensorD<T> &euclidean(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda(x2, y, first_match_dims, last_work_dims, CudaCtx<T>::euclidean, true))
        {
            return y;
        }

        _sum_func([](T xe1, T xe2) -> T
                  { return (xe1 - xe2) * (xe1 - xe2); },
                  x2, y, first_match_dims, last_work_dims);
        y.sqrt(y, last_work_dims);
        return y;
    }


    TensorD<T> &euclidean_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                               TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                               uint first_match_dims = 0, int last_work_dims = 1) const
    {
        if (_run_cuda_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, CudaCtx<T>::euclidean_grad, true))
        {
            return x1_grad;
        }

        return this->_sum_func_grad([](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                                    {
                                        assert(xe2 > 0);
                                        if (Math<float>::almost_zero(ye)){
                                            xe1_grad = xe2_grad = 0;
                                        }
                                        else
                                        {
                                            xe1_grad = (xe1 - xe2) / ye; xe2_grad = - xe1_grad;
                                        } },
                                    x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

#pragma endregion

#pragma region unary_maps
public:
    // linear: y[i] = x[i] * alpha + beta;
    TensorD<T> &linear(TensorD<T> &y, T alpha = 1.0, T beta = 0.0, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, [alpha, beta](const float *x, float *y, uint rows, uint cols)
                      { CudaCtx<T>::linear(x, y, rows, cols, alpha, beta); }))
        {
            return y;
        }

        return _map([alpha, beta](T x) -> T
                    { return x * alpha + beta; },
                    y, last_work_dims);
    }

    TensorD<T> &linear_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                            T alpha = 1.0, T beta = 0.0, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, 
        [alpha, beta](const float *x, const float *y_grad, float *x1_grad, uint rows, uint cols)
                      { CudaCtx<T>::linear_grad(x, y_grad, x1_grad, rows, cols, alpha, beta); }))
        {
            return x1_grad;
        }

        return _map_grad([alpha](T x, T y) -> T
                         { return alpha; },
                         y, y_grad, x1_grad, last_work_dims);
    }

    TensorD<T> &sqrt(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::sqrt))
        {
            return y;
        }

        return _map(Math<T>::sqrt, y, last_work_dims);
    }

    TensorD<T> &sqrt_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                          int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, CudaCtx<T>::sqrt_grad))
        {
            return x1_grad;
        }

        return _map_grad([](T ex, T ey) -> T { return ey != 0 ? 0.5 / ey : 0; },
                         y, y_grad, x1_grad, last_work_dims);
    }

    // y = pow(x + bias, n)
    TensorD<T> &pow(TensorD<T> &y, float n, float bias = 0, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, [n, bias](const float *x, float *y, uint rows, uint cols)
                      { CudaCtx<T>::pow(x, y, rows, cols, n, bias); }))
        {
            return y;
        }

        return _map([n, bias](T ex) -> T
                    { return std::pow(ex + bias, n); },
                    y, last_work_dims);
    }

    TensorD<T> &pow_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                         float n, float bias = 0, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, 
        [n, bias](const float *x, const float *y_grad, float *x1_grad, uint rows, uint cols)
                      { CudaCtx<T>::pow_grad(x, y_grad, x1_grad, rows, cols, n, bias); }))
        {
            return x1_grad;
        }

        return _map_grad([n, bias](T ex, T ey) -> T
                         { return (ex + bias) != 0 ? n * ey / (ex + bias) : 0; },
                         y, y_grad, x1_grad, last_work_dims);
    }

    // y = ln(x + bias)
    TensorD<T> &ln(TensorD<T> &y, float bias = 0, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, [bias](const float *x, float *y, uint rows, uint cols)
                      { CudaCtx<T>::ln(x, y, rows, cols, bias); }))
        {
            return y;
        }

        return _map([bias](T ex) -> T
                    { return std::log(ex + bias); },
                    y, last_work_dims);
    }

    TensorD<T> &ln_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                        float bias = 0, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, 
        [bias](const float *x, const float *y_grad, float *x1_grad, uint rows, uint cols)
                      { CudaCtx<T>::ln_grad(x, y_grad, x1_grad, rows, cols, bias); }))
        {
            return x1_grad;
        }

        return _map_grad([bias](T ex, T ey) -> T
                         { return 1 / (ex + bias); },
                         y, y_grad, x1_grad, last_work_dims);
    }

    // y[i] = exp(x[i] - max(x[i])) / sum(exp(x[i] - max(x[i])))
    TensorD<T> &softmax(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::softmax))
        {
            return y;
        }

        return _map([](const Vector<T> &xv1, Vector<T> &y, uint xv1_start, uint y_start, int last_size) -> void
                    { xv1.softmax(y, xv1_start, y_start, last_size); },
                    y, last_work_dims);
    }

    // note: even subtract by max(x[i]), the grad is still the same
    // d(y/x) = (xdy - ydx) / x^2
    //  dy_j/dx_i = exp(x_j) * -1 * sum(k, exp(x_k))^2 * exp(x_i) = -1 * y_i * y_j if i != j;
    //  dy_i/dx_i = (sum(exp(x_k)) * exp(x_i) - exp(x_i) * exp(x_i)) / sum(exp(x_i))^2 = y_i * (1 - y_i)
    //  dL/dx_i = sum(j, dL/dy_j * dy_j/dx_i) = sum(j, dL/dy_j * dy_j/dx_i or dy_i/dx_i) = sum(j, dL/dy_j * (i == j - y_j) * y_i) =
    //        = y_i * (dL/dy_i - sum(j, dL/dy_j * y_j))
    TensorD<T> &softmax_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y, y_grad, x_grad, last_work_dims, CudaCtx<T>::softmax_grad))
        {
            return x_grad;
        }

        // Jacobian Matrix
        auto map_grad_func = [](const Vector<T> &x, const Vector<T> &y, const Vector<T> &y_grad, Vector<T> &x_grad,
                                uint x_start, uint y_start, uint x_grad_start, uint len) -> void
        {
            T sum = y.dot(y_grad, y_start, y_start, len);
            for (uint i = 0; i < len; ++i)
            {
                x_grad[i + x_grad_start] += y[i + y_start] * (y_grad[i + y_start] - sum);
            }
        };

        return this->_map_grad(map_grad_func, y, y_grad, x_grad, last_work_dims);
    }

    TensorD<T> &activation(Activation_Types type, TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, [type](const float *x, float *y, uint rows, uint cols)
                      { CudaCtx<T>::activation(x, y, rows, cols, type); }))
        {
            return y;
        }

        std::function<T(T)> func;
        switch (type)
        {
        case Activation_Types::None:
        case Activation_Types::Linear:
            func = Math<T>::empty_map;
            break;
        case Activation_Types::Relu:
            func = Math<T>::relu;
            break;
        case Activation_Types::Sigmoid:
            func = Math<T>::sigmoid;
            break;
        case Activation_Types::Tanh:
            func = Math<T>::tanh;
            break;
        }

        return _map(func, y, last_work_dims);
    }

    TensorD<T> &activation_grad(Activation_Types type, const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad,
                                int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x_grad, last_work_dims, 
        [type](const float *x, const float *y_grad, float *x_grad, uint rows, uint cols)
                      { CudaCtx<T>::activation_grad(x, y_grad, x_grad, rows, cols, type); }))
        {
            return x_grad;
        }

        std::function<T(T ex, T ey)> func;
        switch (type)
        {
        case Activation_Types::None:
            break;
        case Activation_Types::Linear:
            func = [](T, T) -> T
            { return 1; };
            break;
        case Activation_Types::Relu:
            func = Math<T>::relu_grad;
            break;
        case Activation_Types::Sigmoid:
            func = Math<T>::sigmoid_grad;
            break;
        case Activation_Types::Tanh:
            func = Math<T>::tanh_grad;
            break;
        }
        return _map_grad(func, y, y_grad, x_grad, last_work_dims);
    }

    // RMSNorm: y = x / sqrt(mean(x^2) + eps)
    TensorD<T> &rms_norm(TensorD<T> &y, float gamma = 1.0f, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, [gamma](const float *x, float *y, uint rows, uint cols)
                      { CudaCtx<T>::rms_norm(x, y, rows, cols, gamma); }))
        {
            return y;
        }

        if (last_work_dims == -1)
            last_work_dims = this->shape();
        assert(last_work_dims <= this->shape());

        TensorD<T> norm, norm1;
        this->norm_ln(norm, 2, last_work_dims);
        norm.pow(norm1, -1, EPSILON, -1.0f);
        uint first_match_dims = this->dim().size() - last_work_dims;
        this->mul(norm1, y, gamma, 0.0f, first_match_dims, 0);

        return y;
    }

    TensorD<T> &rms_norm_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, 
        float gamma = 1.0f, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x_grad, last_work_dims, 
        [gamma](const float *x, const float *y_grad, float *x_grad, uint rows, uint cols)
                      { CudaCtx<T>::rms_norm_grad(x, y_grad, x_grad, rows, cols, gamma); }))
        {
            return x_grad;
        }


        if (x_grad.dim() != this->dim())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        if (last_work_dims == -1)
            last_work_dims = this->shape();
        assert(last_work_dims <= this->shape());
        uint first_match_dims = this->dim().size() - last_work_dims;

        TensorD<T> norm, norm1, y1, norm_grad, norm1_grad;
        this->norm_ln(norm, 2, last_work_dims);
        norm.pow(norm1, -1, EPSILON, -1.0f);
        this->mul(norm1, y1, gamma, 0.0f, first_match_dims, 0);

        this->mul_grad(norm1, y1, y_grad, x_grad, norm1_grad, true, gamma, 0.0f, first_match_dims, 0);
        norm.pow_grad(norm1, norm1_grad, norm_grad, -1, EPSILON, -1.0f);
        this->norm_ln_grad(norm, norm_grad, x_grad, 2, last_work_dims);

        return x_grad;
    }

    // this is RoPE embedding, assume last dim is hidden_dim, last - 1 dim is seq_len
    NEEDS_CUDA TensorD<T> &rope(TensorD<T> &y, double base = 10000.0f) const
    {
        assert(this->dim().size() >= 2);
        uint hidden_dim = this->dim().back();
        uint seq_len = this->dim()[this->dim().size() - 2];
        uint group_size = this->dim_to_size(0, this->shape() - 2);

        Vector<T> freqs(hidden_dim / 2);
        for (uint i = 0; i < hidden_dim / 2; ++i)
        {
            freqs[i] = 1.0f / std::pow(base, 2.0f * i / hidden_dim);
        }

        Vector<T> cos(seq_len * hidden_dim / 2), sin(seq_len * hidden_dim / 2);
        for (uint i = 0; i < seq_len; ++i)
            for (uint j = 0; j < hidden_dim / 2; ++j)
            {
                T theta = i * freqs[j];
                cos[i * hidden_dim/2 + j] = std::cos(theta);
                sin[i * hidden_dim/2 + j] = std::sin(theta);
            }

        y.reset(dim());

        for (uint i = 0; i < group_size; ++i)
        {
            for (uint j = 0; j < seq_len; ++j)
            {
                for (uint k = 0; k < hidden_dim; k += 2)
                {
                    uint offset = i * size() / group_size + j * hidden_dim + k;
                    uint cs_offset = j * hidden_dim  / 2 + k / 2;
                    y[offset] = (*this)[offset] * cos[cs_offset] - (*this)[offset + 1] * sin[cs_offset];
                    y[offset + 1] = (*this)[offset] * sin[cs_offset] + (*this)[offset + 1] * cos[cs_offset];
                }
            }
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> &rope_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, double base = 10000.0f) const
    {
        assert(this->dim().size() >= 2);
        uint hidden_dim = this->dim().back();
        uint seq_len = this->dim()[this->dim().size() - 2];
        uint group_size = this->dim_to_size(0, this->shape() - 2);

        Vector<T> freqs(hidden_dim / 2);
        for (uint i = 0; i < hidden_dim / 2; ++i)
        {
            freqs[i] = 1.0f / std::pow(base, 2.0f * i / hidden_dim);
        }

        Vector<T> cos(seq_len * hidden_dim / 2), sin(seq_len * hidden_dim / 2);
        for (uint i = 0; i < seq_len; ++i)
            for (uint j = 0; j < hidden_dim / 2; ++j)
            {
                T theta = i * freqs[j];
                cos[i * hidden_dim /2 + j] = std::cos(theta);
                sin[i * hidden_dim /2 + j] = std::sin(theta);
            }

        if (x_grad.dim() != this->dim())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        /*
        y_i = x_i * cos - x_i+1 * sin
        y_i+1 = x_i * sin + x_i+1 * cos

        x_grad_i = y_grad_i * cos + y_grad_i+1 * sin
        x_grad_i+1 = y_grad_i * -sin + y_grad_i+1 * cos
        */
        for (uint i = 0; i < group_size; ++i)
        {
            for (uint j = 0; j < seq_len; ++j)
            {
                for (uint k = 0; k < hidden_dim; k += 2)
                {
                    uint offset = i * size() / group_size + j * hidden_dim + k;
                    uint cs_offset = j * hidden_dim / 2 + k / 2;
                    x_grad[offset] += y_grad[offset] * cos[cs_offset] + y_grad[offset + 1] * sin[cs_offset];
                    x_grad[offset + 1] += -y_grad[offset] * sin[cs_offset] + y_grad[offset + 1] * cos[cs_offset];
                }
            }
        }

        return x_grad;
    }

#pragma endregion

#pragma region unary_reduces

public:
    inline TensorD<T> &sum(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::sum, true))
        {
            return y;
        }

        return _sum_func(Math<T>::empty_map, y, last_work_dims);
    }

    inline TensorD<T> &sum_grad(const TensorD<T> &y, const TensorD<T> &y_grad,
                                TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, CudaCtx<T>::sum_grad, true))
        {
            return x1_grad;
        }

        return _sum_func_grad([](T e) -> T
                              { return 1; },
                              y, y_grad, x1_grad, last_work_dims);
    }

    TensorD<T> &avg(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::avg, true))
        {
            return y;
        }

        T n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        // note: this is not optimal since each unit just needs one /, now each elem needs one /.
        return _sum_func([n](T e) -> T
                         { return e / n; },
                         y, last_work_dims);
    }

    TensorD<T> &avg_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y_grad, x1_grad, last_work_dims, CudaCtx<T>::avg_grad, true))
        {
            return x1_grad;
        }

        T n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return _sum_func_grad([n](T e) -> T
                              { return 1 / n; },
                              y, y_grad, x1_grad, last_work_dims);
    }

    // note: this is var, not stdvar
    NEEDS_CUDA TensorD<T> &var(TensorD<T> &y, bool biased = false, int last_work_dims = -1) const
    {
        if (last_work_dims < 0)
            last_work_dims = this->shape();
        uint first_match_dims = this->shape() - last_work_dims;

        TensorD<T> mean;
        avg(mean, last_work_dims);
        TensorD<T> mean_inf;
        mean.inflate(mean_inf, Vector(this->dim().subset(first_match_dims, last_work_dims))); // make each mean value in last dim to be the size of work_size, all the same value
        // TODO: we do this because current reduce & bin_reduce only support matched work_size, here the work_size is n VS 1

        uint n = dim_to_size(0, last_work_dims, false);
        if (biased)
            --n;
        assert(n > 0);
        // note: this is not optimal since each unit just needs one /, now each elem needs one /.

        return _sum_func([n](T e, T m) -> T
                         { return (e - m) * (e - m) / n; },
                         mean_inf, y, first_match_dims, last_work_dims);
    }

    NEEDS_CUDA TensorD<T> &var_grad(const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x_grad, bool biased = false, int last_work_dims = -1) const
    {
        auto reduce_grad_func = [](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            uint n = len;
            float m = x1.avg(x1_start, len);
            // dmean_dx = 1/n
            // dvar_dx = 2/n * (e - m) * (1 - 1/n)
            for (uint i = 0; i < len; ++i)
            {
                float dmean_dx = 1.0 / n;
                float dvar_dx = 2.0 / n * (x1[x1_start + i] - m) * (1 - dmean_dx);
                x1_grad[x1_grad_start + i] += dvar_dx * y_grad;
            }
        };

        return _reduce_grad(reduce_grad_func, y, y_grad, x_grad, last_work_dims);
    }

    TensorD<T> &max(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::max, true))
        {
            return y;
        }

        return _reduce(
            Math<T>::empty_map, [](T res, T e) -> T
            { return e > res ? e : res; },
            y, last_work_dims);
    }

    TensorD<T> &max_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y, y_grad, x1_grad, last_work_dims, CudaCtx<T>::max_grad, true))
        {
            return x1_grad;
        }

        auto reduce_grad_func = [](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            for (uint i = 0; i < len; ++i)
            {
                if (x1[x1_start + i] == y)
                {
                    x1_grad[x1_grad_start + i] += 1 * y_grad;
                }
                else
                {
                    x1_grad[x1_grad_start + i] = 0;
                }
            }
        };

        return _reduce_grad(reduce_grad_func, y, y_grad, x1_grad, last_work_dims);
    }

    TensorD<T> &min(TensorD<T> &y, int last_work_dims = -1) const
    {
        if (_run_cuda(y, last_work_dims, CudaCtx<T>::min, true))
        {
            return y;
        }

        return _reduce(
            Math<T>::empty_map, [](T res, T e) -> T
            { return e < res ? e : res; },
            y, last_work_dims);
    }

    TensorD<T> &min_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                         int last_work_dims = -1) const
    {
        if (_run_cuda_grad(y, y_grad, x1_grad, last_work_dims, CudaCtx<T>::min_grad, true))
        {
            return x1_grad;
        }

        return max_grad(y, y_grad, x1_grad, last_work_dims);
    }

    NEEDS_CUDA TensorD<T> &norm_ln(TensorD<T> &y, float n, int last_work_dims = -1) const
    {
        if (last_work_dims == -1)
            last_work_dims = this->shape();
        assert(last_work_dims <= this->shape());
        uint len = dim_to_size(0, last_work_dims, false);
        assert(len != 0);

        TensorD<T> y0;
        _reduce([n, len](T e) -> T
                { return std::pow(e, n) / len; }, Math<T>::sum_reduce0,
                y0, last_work_dims);

        assert(n != 0);
        y0.pow(y, 1.0f / n);
        return y;
    }

    NEEDS_CUDA TensorD<T> &norm_ln_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, float n, int last_work_dims = -1) const
    {
        if (x_grad.dim() != this->dim())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        if (last_work_dims == -1)
            last_work_dims = this->shape();

        uint len = dim_to_size(0, last_work_dims, false);
        assert(len != 0);

        TensorD<T> y0, y0_grad;

        _reduce([n, len](T e) -> T
                { return std::pow(e, n) / len; }, Math<T>::sum_reduce0,
                y0, last_work_dims);

        assert(n != 0);
        y0.pow_grad(y, y_grad, y0_grad, 1.0f / n);

        auto reduce_grad_func = [n](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            for (uint i = 0; i < len; ++i)
            {
                x1_grad[x1_grad_start + i] += y_grad * n / len * std::pow(x1[x1_start + i], n - 1);
            }
        };

        return _reduce_grad(reduce_grad_func, y0, y0_grad, x_grad, last_work_dims);

        return x_grad;
    }

#pragma endregion

#pragma region update_ops
    /****************************************************************************************************************/
    /* below are tensor update ops
    supported ops:
        set(one elem), set(one span)
        swap_adj/swap/move_forward
        TODO(from pytorch): flatten, reshape(light_reshape), cat, chunk, gather, index_select, split, transpose, squeeze
    */

    /* ideal func is permutation: reorder all data under dims to be a new order [a, b, c, d, e] => [c, a, d, e, b]
       core problem is how to make it optimal in computation complexity and space complexity
       for now the unit ops we supported:
           swap 2 dims in the dim vector, for now all swaps are adjacent dims: [a b x c d y e] => [a b y c d x e]
           move dims to front dims [a b x c d y1 y2 e] => [a b y1 y2 c d x e]
           we could use move_forward or swap to implement any permutation ops, even not optimal
       used by below:
            Linear: [batch_size, yput_dim] => [yput_dim, batch_size] // swap 1 dim at bottom
            Rnn: [node_len, batch_size, yput_dim] => [batch_size, node_len, yput_dim] // swap 1 dim at top
            Conv: [groups, batch_size, y_height, y_width, y_group_count]
                => [batch_size, groups, ...]; // swap first 1 dim
                => [batch_size, y_group_count, groups, y_height, y_width]; // move last dim to front
            Transformer:
                [batch_size, node_len, multi_head, yput_dim] => [batch_size, multi_head, node_len, yput_dim] // swap middle dim
                [batch_size, node_len, multi_head, yput_dim] => [batch_size, multi_head, yput_dim, node_len] // move 2 dims to front
    */
    // swap_adj(1) => [2, 3, 4, 5] => [2, 4, 3, 5]
    /* input as below: represent as original shape
      01234   12345   23456      *****   *****   *****
      56789   67890   78901      *****   *****   *****
      *****   *****   *****      *****   *****   *****
      *****   *****   *****      *****   *****   *****

      yut as below:
      01234   56789   *****   *****      *****   *****   *****   *****
      12345   67890   *****   *****      *****   *****   *****   *****
      23456   78901   *****   *****      *****   *****   *****   *****
    */
    /* input as below: represent as matrix, flatten all the dims >= first_dim + 1
       this is to treat each unit as one elem, and do transpose for this matrix
      01234,56789,**********      ********************
      12345,67890,**********      ********************
      23456,78901,**********      ********************

      yut as below:
      01234,12345,23456,      ********************
      56789,67890,78901,      ********************
      ***************      ********************
      ***************      ********************
    */

    // swap any two dims in the tensor
    // grad is reverse op
    NEEDS_CUDA TensorD<T> &swap(TensorD<T> &y, uint first_dim, uint second_dim) const
    {
        // at least 2 dims
        assert(first_dim != second_dim);
        if (first_dim > second_dim)
            Math<uint>::swap(first_dim, second_dim);
        assert(shape() > 1 && second_dim < shape());
        Vector<uint> y_dim(dim());
        y_dim[first_dim] = dim()[second_dim]; // 5
        y_dim[second_dim] = dim()[first_dim]; // 3
        y.reset(y_dim);

        uint unit_len = dim_to_size(0, shape() - second_dim - 1, false);               // unit move unit, above eg is 7
        uint group_count = dim_to_size(0, first_dim);                                  // 2
        uint group_len = size() / group_count;                                         // 5 * 4*6 * 3 * 7
        uint row_count = y_dim[first_dim];                                             // 5
        uint row_inc = unit_len;                                                       // 7
        uint col_group_count = dim_to_size(first_dim + 1, second_dim - first_dim - 1); // 4*6
        uint col_group_inc = row_count * row_inc;                                      // 5 * 7
        uint col_count = dim()[first_dim];                                             // 3
        uint col_inc = group_len / col_count;                                          // 5 * 4*6 * 7
        uint in_start = 0, y_start = 0;
        for (uint g = 0; g < group_count; ++g) // 2 * 4*6 * 3 * 7
        {
            for (uint r = 0; r < row_count; ++r)              // 5 * 4*6 * 3 * 7
                for (uint cg = 0; cg < col_group_count; ++cg) // 4*6 * 3 * 7
                    for (uint c = 0; c < col_count; ++c)      // 3 * 7
                    {
                        in_start = g * group_len + r * row_inc + cg * col_group_inc + c * col_inc; // TODO: perf: change * to +
                        y.vector().set(y_start, this->vector(), in_start, unit_len);
                        y_start += unit_len;
                    }
        }

        return y;
    }

    // grad is reverse op
    NEEDS_CUDA TensorD<T> &move_forward(TensorD<T> &y, uint move_from, uint move_len, uint move_to) const
    {
        assert(move_len > 0);
        assert(move_from + move_len <= shape());
        assert(move_from > move_to);

        Vector<uint> y_dim(dim());
        y_dim.move_forward(move_from, move_len, move_to);
        y.reset(y_dim);

        // move_forward({1, 2,3,4,5, 6}, 3,2, 1) => {1, 4, 5, 2, 3, 6}
        uint unit_len = dim_to_size(0, shape() - move_from - move_len, false); // unit move unit, above eg is 6
        uint group_count = dim_to_size(0, move_to);                            // 1
        uint group_len = size() / group_count;                                 // 4*5*2*3*6
        uint row_count = dim_to_size(move_from, move_len);                     // 4*5, row count is the dims moved to front
        uint row_inc = unit_len;                                               // 6
        uint col_count = dim_to_size(move_to, move_from - move_to);            // 2*3
        uint col_inc = row_count * row_inc;                                    // 4*5*6
        uint in_start = 0, y_start = 0;
        for (uint g = 0; g < group_count; ++g)
        {
            for (uint r = 0; r < row_count; ++r)
                for (uint c = 0; c < col_count; ++c)
                {
                    in_start = g * group_len + r * row_inc + c * col_inc; // TODO: perf: change * to +
                    y.vector().set(y_start, this->vector(), in_start, unit_len);
                    y_start += unit_len;
                }
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> &im2col(TensorD<T> &y, uint groups = 1, uint kernel_x = 1, uint kernel_y = 1,
                       uint stride_x = 1, uint stride_y = 1, uint padding_x = 0, uint padding_y = 0) const
    {
        assert(this->shape() == 4);
        assert(kernel_x > 0 && kernel_y > 0);
        assert(stride_x > 0 && stride_y > 0);
        uint batch_size = dim()[0], in_channels = dim()[1], in_height = dim()[2], in_width = dim()[3];
        // after padding, all the elements are used exactly
        assert((in_width + padding_x * 2 - kernel_x) % stride_x == 0);
        assert((in_height + padding_y * 2 - kernel_y) % stride_y == 0);
        uint in_channels_per_group = in_channels / groups;
        uint out_height = (in_height + padding_y * 2 - kernel_y) / stride_y + 1;
        uint out_width = (in_width + padding_x * 2 - kernel_x) / stride_x + 1;
        // note: no automatic padding
        // uint out_height = (in_height + padding_y * 2) / stride_y + (in_height + padding_y * 2) % stride_y != 0;
        // uint out_width = (in_width + padding_x * 2) / stride_x + (in_width + padding_x * 2) % stride_x != 0;

        y.reset({groups, batch_size, out_height, out_width, in_channels_per_group, kernel_y, kernel_x}, TensorInit_Types::Zero);
        uint col_i = 0;

        for (uint group_i = 0; group_i < groups; ++group_i)
            for (uint image_i = 0; image_i < batch_size; ++image_i)
                for (uint out_y = 0; out_y < out_height; ++out_y)
                    for (uint out_x = 0; out_x < out_width; ++out_x)
                    {
                        // this is one row of col
                        uint start_x = stride_x * out_x, start_y = stride_y * out_y; // left/top corner in input image for this scan
                        for (uint ingroup_c = 0; ingroup_c < in_channels_per_group; ++ingroup_c)
                        {
                            uint c = group_i * in_channels_per_group + ingroup_c;
                            uint offset_c = image_i * in_channels * in_height * in_width + c * in_height * in_width;
                            for (uint ky = 0; ky < kernel_y; ++ky)
                            {
                                int scan_y = start_y + ky - padding_y; // may be neg
                                int offset_y = offset_c + scan_y * in_width;
                                for (uint kx = 0; kx < kernel_x; ++kx)
                                {
                                    int scan_x = start_x + kx - padding_x;
                                    float v = 0; // default padding value
                                    if (scan_x >= 0 && scan_x < in_width && scan_y >= 0 && scan_y < in_height)
                                    {
                                        y[col_i] = this->vector()[offset_y + scan_x];
                                    }

                                    col_i++;
                                }
                            }
                        }
                    }
        return y;
    }

    // note: totally same code as im2_col, only one line difference in the end
    NEEDS_CUDA TensorD<T> &im2col_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad,
                            uint groups, uint kernel_x, uint kernel_y,
                            uint stride_x, uint stride_y, uint padding_x, uint padding_y) const
    {
        assert(this->shape() == 4);
        assert(kernel_x > 0 && kernel_y > 0);
        assert(stride_x > 0 && stride_y > 0);
        uint batch_size = dim()[0], in_channels = dim()[1], in_height = dim()[2], in_width = dim()[3];
        // after padding, all the elements are used exactly
        assert((in_width + padding_x * 2 - kernel_x) % stride_x == 0);
        assert((in_height + padding_y * 2 - kernel_y) % stride_y == 0);
        uint in_channels_per_group = in_channels / groups;
        uint out_height = (in_height + padding_y * 2 - kernel_y) / stride_y + 1;
        uint out_width = (in_width + padding_x * 2 - kernel_x) / stride_x + 1;
        // note: no automatic padding
        // uint out_height = (in_height + padding_y * 2) / stride_y + (in_height + padding_y * 2) % stride_y != 0;
        // uint out_width = (in_width + padding_x * 2) / stride_x + (in_width + padding_x * 2) % stride_x != 0;

        // y.reset({groups, batch_size, out_height, out_width, in_channels_per_group, kernel_y, kernel_x});
        if (x_grad.dim() != this->dim())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        uint col_i = 0;

        for (uint group_i = 0; group_i < groups; ++group_i)
            for (uint image_i = 0; image_i < batch_size; ++image_i)
                for (uint out_y = 0; out_y < out_height; ++out_y)
                    for (uint out_x = 0; out_x < out_width; ++out_x)
                    {
                        // this is one row of col
                        uint start_x = stride_x * out_x, start_y = stride_y * out_y; // left/top corner in input image for this scan
                        for (uint ingroup_c = 0; ingroup_c < in_channels_per_group; ++ingroup_c)
                        {
                            uint c = group_i * in_channels_per_group + ingroup_c;
                            uint offset_c = image_i * in_channels * in_height * in_width + c * in_height * in_width;
                            for (uint ky = 0; ky < kernel_y; ++ky)
                            {
                                int scan_y = start_y + ky - padding_y; // may be neg
                                int offset_y = offset_c + scan_y * in_width;
                                for (uint kx = 0; kx < kernel_x; ++kx)
                                {
                                    int scan_x = start_x + kx - padding_x;
                                    float v = 0; // default padding value
                                    if (scan_x >= 0 && scan_x < in_width && scan_y >= 0 && scan_y < in_height)
                                    {
                                        x_grad[offset_y + scan_x] += y_grad[col_i];
                                    }

                                    col_i++;
                                }
                            }
                        }
                    }
        return x_grad;
    }

    NEEDS_CUDA void divide(TensorDArray<T> &y, uint first_match_dims = 1) const
    {
        assert(first_match_dims <= shape());
        if (first_match_dims == 0)
        {
            y.reserve(1);
            y[0].deep_copy(*this);
        }
        else
        {
            uint divide_count = dim_to_size(0, first_match_dims);
            Vector<uint> sub_tensor_dim(dim().subset(first_match_dims, shape() - first_match_dims));
            uint sub_tensor_size = this->size() / divide_count;
            y.reserve(divide_count);
            for (uint i = 0; i < divide_count; ++i)
            {
                y[i].reset(sub_tensor_dim);
                y[i].vector().set(0, this->vector(), i * sub_tensor_size, sub_tensor_size);
            }
        }
    }

    NEEDS_CUDA void divide_grad(const TensorDArray<T> &y, const TensorDArray<T> &y_grad, TensorD<T> &x_grad, uint first_match_dims = 1) const
    {
        assert(first_match_dims <= shape());
        if (x_grad.dim() != dim())
        {
            x_grad.reset(dim(), TensorInit_Types::Zero);
        }

        for (uint i = 0; i < y_grad.size(); ++i)
        {
            x_grad.vector().set(i * y_grad[i].size(), y_grad[i].vector(), 0, y_grad[i].size());
        }
    }

    NEEDS_CUDA static TensorD<T> combine(const TensorDArray<T> &x, const Vector<uint> &first_dims = {})
    {
        TensorD<T> y;

        if (x.size() == 0)
            return y;
        uint size = x[0].vector().size();

        Vector<uint> y_dim = x[0].dim();
        if (first_dims.size() == 0)
        {
            y_dim.insert(0, {x.size()});
        }
        else
        {
            assert(first_dims.product() == x.size());
            y_dim.insert(0, first_dims);
        }

        y.reset(y_dim);
        y.vector().reserve(size * x.size());
        for (uint i = 0; i < x.size(); ++i)
        {
            assert(x[i].vector().size() == size);
            y.vector().set(i * size, x[i].vector());
        }

        return y;
    }

    NEEDS_CUDA static void Combine_Grad(const TensorDArray<T> &x, const TensorD<T> &y, const TensorD<T> &y_grad,
                             TensorDArray<T> &x_grad, const Vector<uint> &first_dims = {})
    {
        uint size = x.size();
        if (size == 0)
            return;

        if (x_grad.size() == 0)
        {
            x_grad.reserve(size);
        }

        uint sub_tensor_size = y.size() / size;

        for (uint i = 0; i < size; ++i)
        {
            assert(sub_tensor_size == x[i].size());
            x_grad[i].reset(x[i].dim());
            x_grad[i].vector().set(0, y_grad.vector(), i * sub_tensor_size, sub_tensor_size);
        }
    }

    // this is virtual tensor, shared same data with source Tensor
    // TODO: change this to reshape
    TensorD<T> &merge_dim(TensorD<T> &y, uint from, uint len) const
    {
        assert(from + len <= shape());
        Vector<uint> dims;
        for (uint i = 0; i < from; ++i)
        {
            dims.push_back(this->dim()[i]);
        }

        uint merged = 1;
        for (uint i = from; i < from + len; ++i)
        {
            merged *= this->dim()[i];
        }

        dims.push_back(merged);
        for (uint i = from + len; i < shape(); ++i)
        {
            dims.push_back(this->dim()[i]);
        }

        y._dim.copy(dims);

        y.vector_ptr() = this->vector_ptr();

        return y;
    }

    TensorD<T> &merge_dim_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, uint from, uint len) const
    {
        if (x_grad.shape() != this->shape())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().copy(y_grad.vector());
        return x_grad;
    }

    NEEDS_CUDA TensorD<T> &inflate(TensorD<T> &y, const Vector<uint> &dims = {}) const
    {
        if (dims.size() == 0)
        {
            y.deep_copy(*this);
            return y;
        }

        uint inflation_size = dims.product();
        assert(inflation_size > 0);
        Vector<uint> dim = this->dim();
        if (dims.size() > 0)
            dim.append(dims);
        y.reset(dim);
        for (uint i = 0; i < this->size(); ++i)
        {
            y.vector().set_each((*this)[i], i * inflation_size, inflation_size);
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> &inflate_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, const Vector<uint> &dims = {}) const
    {
        if (dims.size() == 0)
        {
            x_grad.deep_copy(y_grad);
            return x_grad;
        }

        uint inflation_size = dims.product();

        assert(inflation_size > 0);
        assert(y_grad.size() / inflation_size == size());

        if (!x_grad.dim().equals_to(dim()))
        {
            x_grad.reset(dim(), TensorInit_Types::Zero);
        }

        y_grad.sum(x_grad, 1);
        return x_grad;
    }

    // note: shared data, should destroy original reference
    TensorD<T> &squeeze(TensorD<T> &y, int dim = -1) const
    {
        Vector<uint> new_dim;
        for (uint i = 0; i < this->dim().size(); ++i)
        {
            if (this->dim()[i] != 1 || dim >= 0 && i != dim)
            {
                new_dim.push_back(this->dim()[i]);
            }
        }

        y.weak_copy(*this);
        y._dim.copy(new_dim);
        return y;
    }

    TensorD<T> &squeeze_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, int dim = -1) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().add(y_grad.vector(), x_grad.vector());
        return x_grad;
    }

    TensorD<T> unsqueeze(uint dim) const
    {
        assert(dim <= this->dim().size());
        Vector<uint> new_dim = this->dim();
        new_dim.insert(dim, 1);

        TensorD<T> y;
        y.weak_copy(*this);
        y._dim.copy(new_dim);
        return y;
    }

    TensorD<T> &unsqueeze_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, uint dim) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().add(y_grad.vector(), x_grad.vector());
        return x_grad;
    }

    TensorD<T> reshape(const Vector<uint> &dim) const
    {
        assert(dim.product() == this->size());
        TensorD<T> y;
        y.reset(dim);
        y.vector().copy(this->vector());
        return y;
    }

    TensorD<T> &reshape_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, const Vector<uint> &dim) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().add(y_grad.vector(), x_grad.vector());
        return x_grad;
    }

    // copy subset of data from y, starting from offset, copying size of dim.product()
    NEEDS_CUDA TensorD<T> &subset(TensorD<T> &y, const Vector<uint> &dim, uint offset) const
    {
        uint size = dim.product();
        assert(offset + size <= this->size());

        y.reset(dim);
        y.vector().copy(this->vector(), offset, size); // deep copy
        return y;
    }

    SUGAR TensorD<T> subset(const Vector<uint> &dim, uint offset) const
    {
        TensorD<T> y;
        return subset(y, dim, offset);
    }

    NEEDS_CUDA TensorD<T> &subset_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, const Vector<uint> &dim, uint offset) const
    {
        uint size = dim.product();
        assert(offset + size <= this->size());

        if (!x_grad.dim().equals_to(this->dim()))
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().set(offset, y_grad.vector(), 0, size);

        return x_grad;
    }

    NEEDS_CUDA TensorD<T> &dropout(TensorD<T> &y, float p) const
    {
        y.deep_copy(*this);
        y.vector().dropout(y.vector(), p);
        return y;
    }

    NEEDS_CUDA TensorD<T> &dropout_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, float p) const
    {
        if (x_grad.dim() != this->dim())
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().add(y_grad.vector(), x_grad.vector());
        return x_grad;
    }

    // this is to merge several dims
    // e.g., merge(1, 1), [batch_size, multi_head, node_len, output_dim] => [batch_size, node_len, output_dim]
    /* inputs: [2, 2, 4, 3]
    0,0,0, 1,1,1, 2,2,2, 3,3,3,
    0,0,0, 1,1,1, 2,2,2, 3,3,3,

    0,0,0, 1,1,1, 2,2,2, 3,3,3,
    0,0,0, 1,1,1, 2,2,2, 3,3,3,

    outputs: [2, 4, 3]
    0,0,0, 2,2,2, 4,4,4, 6,6,6,

    0,0,0, 2,2,2, 4,4,4, 6,6,6,
    */
    // TODO: we can implement this faster by not calling move_forward & sum
    /*NOTEST TensorD<T> &merge(TensorD<T>& y, uint from, uint len) const{
    // so far use move_forward & sum to implement
        TensorD<T> z;
        this->move_forward(z, from, len, this->shape());
        return z.sum(y, len);
    }*/

    // use-case-2: {batch_size, node_len}.encode{batch_size, node_len, dict_size} => {batch_size, node_len} => get target prob
    //    encode__(x2, y, 2)

    // [x, y, z, ...].encode({dict_size, input_dim}) => [x, y, z, ..., input_dim]
    NEEDS_CUDA TensorD<T> &encode_by_dict(const TensorD<T> &dict, TensorD<T> &y) const
    {
        assert(dict.shape() == 2);
        uint dict_size = dict.dim()[0];
        uint input_dim = dict.dim()[1];

        Vector<uint> new_dim = this->dim();
        new_dim.push_back(input_dim);
        y.reset(new_dim);

        for (uint i = 0; i < this->size(); ++i)
        {
            uint dict_id = this->vector()[i];
            assert(dict_id < dict_size);
            y.vector().add_(dict.vector(), i * input_dim, dict_id * input_dim, input_dim);
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> &encode_by_dict_grad(const TensorD<T> &dict, const TensorD<T> &y, const TensorD<T> &y_grad,
                                    TensorD<T> &dict_grad) const
    {
        assert(dict.shape() == 2);
        uint dict_size = dict.dim()[0];
        uint input_dim = dict.dim()[1];

        if (!dict_grad.dim().equals_to(dict.dim()))
        {
            dict_grad.reset(dict.dim(), TensorInit_Types::Zero);
        }

        for (uint i = 0; i < this->size(); ++i)
        {
            uint dict_id = this->vector()[i];
            assert(dict_id < dict_size);
            dict_grad.vector().add_(y_grad.vector(), dict_id * input_dim, i * input_dim, input_dim);
        }

        return dict_grad;
    }

    // use-case-2: {batch_size, node_len}.encode{batch_size, node_len, dict_size} => {batch_size, node_len} => get target prob
    //    encode__(x2, y, 2)

    // [x, y, z, ...].search({x, y, z, ..., dict_size}) => [x, y, z, ...]
    NEEDS_CUDA TensorD<T> &search_by_dict(const TensorD<T> &dict, TensorD<T> &y, int padding_id = -1) const
    {
        assert(dict.shape() == this->shape() + 1);
        assert(dict.dim().match_bottom(this->dim(), this->shape(), false));
        uint dict_size = dict.dim()[dict.shape() - 1];

        y.reset(this->dim());

        for (uint i = 0; i < this->size(); ++i)
        {
            uint dict_id = this->vector()[i];
            assert(dict_id < dict.size());
            if (padding_id >= 0 && dict_id == padding_id)
                y[i] = 0; // no confidence for padding
            else
                y[i] = dict[i * dict_size + dict_id];
        }

        return y;
    }

    // [x, y, z, ...].search({x, y, z, ..., dict_size}) => [x, y, z, ...]
    NEEDS_CUDA TensorD<T> &search_by_dict_grad(const TensorD<T> &dict, const TensorD<T> &y, const TensorD<T> &y_grad,
                                    TensorD<T> &dict_grad, int padding_id = -1) const
    {
        assert(dict.shape() == this->shape() + 1);
        uint dict_size = dict.dim()[dict.shape() - 1];

        if (!dict_grad.dim().equals_to(dict.dim()))
        {
            dict_grad.reset(dict.dim(), TensorInit_Types::Zero);
        }

        for (uint i = 0; i < this->size(); ++i)
        {
            uint dict_id = this->vector()[i];
            assert(dict_id < dict_size);
            if (padding_id >= 0 && dict_id == padding_id)
            {
                dict_grad.vector()[padding_id * dict_size] = 0;
            }
            else
            {
                dict_grad.vector()[i * dict_size + dict_id] = y_grad[i];
            }
        }

        return dict_grad;
    }

    // this is convert the id with max value in dict_size to be id
    // treat the last dim size as dict_size, and find the max value of the dim, and assign the id with max value
    // last dim will reduce to one scalar value
    NEEDS_CUDA TensorD<T> &decode__(TensorD<T> &y) const
    {
        assert(this->shape() >= 1);
        uint dict_size = this->dim()[this->shape() - 1];
        y.reset(Vector(this->dim().subset(0, this->shape() - 1)));
        if (y.shape() == 0)
        {
            y.reset({1});
        }

        for (uint i = 0; i < y.size(); ++i)
        {
            T max = this->vector()[i * dict_size];
            uint max_id = 0;
            for (uint j = 1; j < dict_size; ++j)
            {
                T value = this->vector()[i * dict_size + j];
                if (value > max)
                {
                    max = value;
                    max_id = j;
                }
            }

            y[i] = max_id;
        }

        return y;
    }

    NEEDS_CUDA void append(const TensorD<T> &x2, TensorD<T> &y, uint dim_to_inc = 0) const
    {
        y.deep_copy(*this);
        if (this->shape() > 0)
        {
            assert(dim_to_inc + x2.shape() + 1 == this->shape());
            y.vector().append(x2.vector());
            y.dim()[dim_to_inc] += 1;
        }
        else
        {
            assert(dim_to_inc == 0);
            y.deep_copy(x2, 1);
            y.dim().insert(0, 1);
        }
    }

    NEEDS_CUDA void append_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                     TensorD<T> &x2_grad, bool enable_x2_grad = true, uint dim_to_inc = 0) const
    {
        if (!x1_grad.dim().equals_to(this->dim()))
            x1_grad.reset(this->dim(), TensorInit_Types::Zero);

        if (enable_x2_grad && !x2_grad.dim().equals_to(x2.dim()))
            x2_grad.reset(x2.dim(), TensorInit_Types::Zero);

        if (this->shape() > 0)
        {
            assert(dim_to_inc + x2.shape() + 1 == this->shape());
            x1_grad.vector().set(0, y_grad.vector(), 0, x1_grad.size());
            if (enable_x2_grad)
                x2_grad.vector().set(0, y_grad.vector(), x1_grad.size(), x2_grad.size());
        }
        else
        {
            assert(dim_to_inc == 0);
            if (enable_x2_grad)
                x2_grad.vector().set(0, y_grad.vector(), x1_grad.size(), x2_grad.size());
        }
    }

    TensorD<T> &map(TensorD<T> &y, const std::function<float(float)> &func) const
    {
        y.reset(this->dim());
        this->vector().map(func, y.vector());

        return y;
    }

    TensorD<T> &map_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad,
                         const std::function<float(float)> &func_grad) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
            x_grad.reset(this->dim(), TensorInit_Types::Zero);

        TensorD<T> x_grad_tmp(this->dim(), TensorInit_Types::Zero);
        TensorD<T> x_grad_tmp1(this->dim(), TensorInit_Types::Zero);
        TensorD<T> x_grad_tmp2(this->dim(), TensorInit_Types::Zero);
        this->vector().map(func_grad, x_grad_tmp.vector());

        y_grad.dot(x_grad_tmp, x_grad_tmp1, this->shape(), 0);

        x_grad.add(x_grad_tmp1, x_grad_tmp2, 1, 1, 0, this->shape(), 0);

        x_grad.deep_copy(x_grad_tmp2);

        return x_grad;
    }

    TensorD<T> &map(TensorD<T> &y,
                    const std::function<void(const Vector<float> &, uint, uint, Vector<float> &)> &func, uint first_match_dims = 0) const
    {
        y.reset(this->dim());
        assert(first_match_dims <= this->shape());
        uint group_count = this->dim_to_size(0, first_match_dims);
        uint group_size = this->size() / group_count;
        for (uint i = 0; i < group_count; ++i)
        {
            uint group_start = i * group_size;
            uint group_end = group_start + group_size;
            func(this->vector(), group_start, group_size, y.vector());
        }

        return y;
    }

    TensorD<T> &map_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad,
                         const std::function<void(const Vector<float> &, uint, uint, const Vector<float> &, Vector<float> &)> &func_grad, uint first_match_dims = 0) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
            x_grad.reset(this->dim(), TensorInit_Types::Zero);

        uint group_count = this->dim_to_size(0, first_match_dims);
        uint group_size = this->size() / group_count;
        for (uint i = 0; i < group_count; ++i)
        {
            uint group_start = i * group_size;
            uint group_end = group_start + group_size;
            func_grad(this->vector(), group_start, group_size, y_grad.vector(), x_grad.vector());
        }

        return x_grad;
    }

    NEEDS_CUDA Array<TensorD<T>> where(CompareTypes type, T v) const
    {
        uint dim_size = this->dim().size();
        Array<Vector<T>> res(dim_size);
        for (uint i = 0; i < this->size(); ++i)
        {
            T item = (*this)[i];
            bool match = false;
            switch (type)
            {
            case CompareTypes::Equal:
                match = item == v;
                break;
            case CompareTypes::Not_Equal:
                match = item != v;
                break;
            case CompareTypes::Greater_Than:
                match = item > v;
                break;
            case CompareTypes::Greater_Equal_Than:
                match = item >= v;
                break;
            case CompareTypes::Less_Than:
                match = item < v;
                break;
            case CompareTypes::Less_Equal_Then:
                match = item <= v;
                break;
            default:
                assert(false);
                break;
            }

            if (match)
            {
                uint start = i;
                for (uint d = 0; d < dim_size; ++d)
                {
                    uint curr_dim = this->dim()[dim_size - d - 1];
                    assert(curr_dim > 0);
                    res[dim_size - d - 1].push_back(start % curr_dim);
                    start = start / curr_dim;
                }
            }
        }

        Array<TensorD<T>> final_res(dim_size);
        for (uint i = 0; i < dim_size; ++i)
        {
            final_res[i].reset({res[i].size()}, res[i]);
        }

        return final_res;
    }

    // Returns k indices of largest elements along the last dimension
    // Returns a tensor of indices of shape (..., k)
    NEEDS_CUDA CURSOR Array<TensorD<T>> topk(uint k) const
    {
        assert(k > 0);
        assert(this->dim().size() > 0);
        uint last_dim = this->dim().back();
        assert(k <= last_dim);

        uint group_size = this->size() / last_dim;
        Vector<T> group_data(last_dim);
        Vector<uint> indices(k);
        Vector<T> values(k);
        Vector<T> indices_data;
        Vector<T> values_data;
        indices_data.reserve(group_size * k);
        values_data.reserve(group_size * k);

        // Process each group
        for (uint g = 0; g < group_size; g++)
        {
            // Copy current group
            for (uint i = 0; i < last_dim; i++)
            {
                group_data[i] = (*this)[g * last_dim + i];
            }

            // Find top k indices and values
            for (uint i = 0; i < k; i++)
            {
                T max_val = group_data[0];
                uint max_idx = 0;
                for (uint j = 1; j < last_dim; j++)
                {
                    if (group_data[j] > max_val)
                    {
                        max_val = group_data[j];
                        max_idx = j;
                    }
                }
                indices[i] = max_idx;
                values[i] = max_val;
                group_data[max_idx] = std::numeric_limits<T>::lowest();
            }

            // Store results
            for (uint i = 0; i < k; i++)
            {
                indices_data[g * k + i] = indices[i];
                values_data[g * k + i] = values[i];
            }
        }

        // Create output tensors
        Vector<uint> out_dims = this->dim();
        out_dims[out_dims.size() - 1] = k;
        Array<TensorD<T>> result(2);
        result[0].reset(out_dims, indices_data);
        result[1].reset(out_dims, values_data);
        return result;
    }

    // notes: no grad for indices, there are grads for x
    NEEDS_CUDA CURSOR TensorD<T> &topk_grad(const TensorD<T> &y0, const TensorD<T> &y1_grad, TensorD<T> &x_grad, uint k) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
            x_grad.reset(this->dim(), TensorInit_Types::Zero);

        uint group_size = this->size() / this->dim().back();
        for (uint g = 0; g < group_size; g++)
        {
            for (uint i = 0; i < k; i++)
            {
                uint id = y0[g * k + i];
                x_grad[g * this->dim().back() + id] += y1_grad[g * k + i];
            }
        }

        return x_grad;
    }

    NEEDS_CUDA TensorD<T> index(const Array<TensorD<T>> &indices, bool cross = false) const
    {
        assert(indices.size() > 0);
        assert(indices.size() <= this->dim().size());
        uint indices_tuple_size = indices[0].size();
        for (uint i = 1; i < indices.size(); i++)
        {
            assert(indices[i].dim().size() == 1);
            if (cross)
                indices_tuple_size *= indices[i].dim()[0];
            else
                assert(indices[i].size() == indices_tuple_size);
        }

        uint value_size = this->dim_to_size(indices.size());
        Vector<T> values(indices_tuple_size * value_size);
        for (uint i = 0; i < indices_tuple_size; i++)
        {
            Vector<uint> per_dim_ids(indices.size());
            uint start_i = i;
            for (int j = indices.size() - 1; j >= 0; --j)
            {
                if (cross)
                {
                    per_dim_ids[j] = indices[j][start_i % indices[j].dim()[0]];
                    start_i = start_i / indices[j].dim()[0];
                }
                else
                {
                    per_dim_ids[j] = indices[j][i];
                }
            }

            uint this_start = 0;
            for (uint j = 0; j < indices.size(); ++j)
            {
                this_start += per_dim_ids[j] * this->dim_to_size(j + 1);
            }

            for (uint j = 0; j < value_size; ++j)
            {
                values[i * value_size + j] = (*this)[this_start + j];
            }
        }

        Vector<uint> new_dim(this->dim().subset(indices.size()).insert(0, indices_tuple_size));
        TensorD<T> y(new_dim, values);
        return y;
    }

    // notes: no grad for indices, there are grads for x
    NEEDS_CUDA CURSOR TensorD<T> &index_grad(const Array<TensorD<T>> &indices, const TensorD<T> &y_grad, TensorD<T> &x_grad,
                                  bool cross = false) const
    {
        assert(indices.size() <= this->dim().size());
        if (!x_grad.dim().equals_to(this->dim()))
            x_grad.reset(this->dim(), TensorInit_Types::Zero);

        uint indices_tuple_size = indices[0].size();
        for (uint i = 1; i < indices.size(); i++)
        {
            assert(indices[i].dim().size() == 1);
            if (cross)
                indices_tuple_size *= indices[i].dim()[0];
            else
                assert(indices[i].size() == indices_tuple_size);
        }

        uint value_size = this->dim_to_size(indices.size());

        for (uint i = 0; i < indices_tuple_size; i++)
        {
            Vector<uint> per_dim_ids(indices.size());
            uint start_i = i;
            for (int j = indices.size() - 1; j >= 0; --j)
            {
                if (cross)
                {
                    per_dim_ids[j] = indices[j][start_i % indices[j].dim()[0]];
                    start_i = start_i / indices[j].dim()[0];
                }
                else
                {
                    per_dim_ids[j] = indices[j][i];
                }
            }

            uint this_start = 0;
            for (uint j = 0; j < indices.size(); ++j)
            {
                this_start += per_dim_ids[j] * this->dim_to_size(j + 1);
            }

            for (uint j = 0; j < value_size; ++j)
            {
                x_grad[this_start + j] += y_grad[i * value_size + j];
            }
        }

        return x_grad;
    }

    // note: only support 1D indices
    NEEDS_CUDA TensorD<T> assign(const TensorD<T> &values, const TensorD<T> &indices) const
    {
        assert(indices.dim().size() == 1);
        assert(values.dim().size() > 1);
        assert(indices.dim()[0] == values.dim()[0]);
        assert(this->dim().size() == values.dim().size());
        assert(this->dim().subset(1).equals_to(values.dim().subset(1)));

        TensorD<T> y;
        y.deep_copy(*this);

        uint value_size = values.dim_to_size(1);

        for (uint i = 0; i < indices.size(); i++)
        {
            uint this_start = indices[i] * value_size;
            for (uint j = 0; j < value_size; j++)
            {
                y[this_start + j] = values[i * value_size + j];
            }
        }

        return y;
    }

    // note: only support 1D indices
    NEEDS_CUDA TensorD<T> &assign_grad(const TensorD<T> &values, const TensorD<T> &indices, const TensorD<T> &y_grad, TensorD<T> &values_grad) const
    {
        assert(indices.dim().size() == 1);
        assert(values.dim().size() > 1);
        assert(indices.dim()[0] == values.dim()[0]);
        assert(this->dim().size() == values.dim().size());
        assert(this->dim().subset(1).equals_to(values.dim().subset(1)));

        if (values_grad.size() == 0)
            values_grad.reset(values.dim(), TensorInit_Types::Zero);

        uint value_size = values.dim_to_size(1);
        for (uint i = 0; i < indices.size(); i++)
        {
            uint this_start = indices[i] * value_size;
            for (uint j = 0; j < value_size; j++)
            {
                values_grad[i * value_size + j] += y_grad[this_start + j];
            }
        }

        return values_grad;
    }

    NEEDS_CUDA TensorD<T> replace(T cond_value, T if_value, T else_value) const
    {
        TensorD<T> y;

        map(y, [cond_value, if_value, else_value](T x)
            { return x == cond_value ? if_value : else_value; });
        return y;
    }

    NEEDS_CUDA TensorD<T> replace_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, T cond_value, T if_value, T else_value) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
            x_grad.reset(dim(), TensorInit_Types::Zero);

        map_grad(y, y_grad, x_grad, [cond_value, if_value, else_value](T x)
                 { return 1.0f; });

        return x_grad;
    }

    NEEDS_CUDA TensorD<T> insert(uint pos, T value, int last_work_dim = -1) const
    {
        TensorD<T> y(this->dim());
        if (last_work_dim == -1)
            last_work_dim = this->dim().size();
        uint group_size = this->dim_to_size(0, this->dim().size() - last_work_dim);
        uint work_size = this->dim_to_size(0, last_work_dim, false);
        for (uint i = 0; i < group_size; i++)
        {
            for (uint j = 0; j < work_size; j++)
            {
                if (j < pos)
                    y[i * work_size + j] = (*this)[i * work_size + j];
                else if (j == pos)
                    y[i * work_size + j] = value;
                else
                    y[i * work_size + j] = (*this)[i * work_size + j - 1];
            }
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> insert_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, uint pos, T value, int last_work_dim = -1) const
    {
        if (x_grad.dim().size() != this->dim().size())
            x_grad.reset(this->dim(), TensorInit_Types::Zero);

        if (last_work_dim == -1)
            last_work_dim = this->dim().size();
        uint group_size = this->dim_to_size(0, this->dim().size() - last_work_dim);
        uint work_size = this->dim_to_size(0, last_work_dim, false);

        for (uint i = 0; i < group_size; i++)
        {
            for (uint j = 0; j < work_size; j++)
            {
                if (j < pos)
                    x_grad[i * work_size + j] += y_grad[i * work_size + j];
                else if (j == pos)
                    x_grad[i * work_size + j] += y_grad[i * work_size + j];
                else if (j < work_size - 1)
                    x_grad[i * work_size + j] += y_grad[i * work_size + j + 1];
            }
        }

        return x_grad;
    }

    NEEDS_CUDA TensorD<T> merge(const TensorD<T>& x2, TensorD<T>& y, uint dim) const
    {
        assert(dim < this->dim().size());
        assert(this->dim().subset(0, dim).equals_to(x2.dim().subset(0, dim)));
        assert(this->dim().subset(dim + 1).equals_to(x2.dim().subset(dim + 1)));

        Vector<uint> new_dim = this->dim();
        new_dim[dim] += x2.dim()[dim];
        y.reset(new_dim);

        uint group_size = this->dim_to_size(0, dim);
        uint work_size = this->dim_to_size(dim + 1);
        uint x1_offset = 0, x2_offset = 0, y_offset = 0;
        for (uint i = 0; i < group_size; i++)
        {
            for (uint j = 0; j < this->dim()[dim]; ++j)
            {
                for (uint k = 0; k < work_size; k++)
                {
                    y[y_offset++] = (*this)[x1_offset++];
                }
            }

            for (uint j = 0; j < x2.dim()[dim]; ++j)
            {
                for (uint k = 0; k < work_size; k++)
                {
                    y[y_offset++] = x2[x2_offset++];
                }
            }
        }

        return y;
    }

    NEEDS_CUDA TensorD<T> merge_grad(const TensorD<T>& x2, const TensorD<T>& y, const TensorD<T>& y_grad, TensorD<T>& x1_grad, TensorD<T>& x2_grad, uint dim) const
    {
        assert(dim < this->dim().size());
        assert(this->dim().subset(0, dim).equals_to(x2.dim().subset(0, dim)));
        assert(this->dim().subset(dim + 1).equals_to(x2.dim().subset(dim + 1)));

        if (!x1_grad.dim().equals_to(this->dim()))
            x1_grad.reset(this->dim(), TensorInit_Types::Zero);

        if (!x2_grad.dim().equals_to(x2.dim()))
            x2_grad.reset(x2.dim(), TensorInit_Types::Zero);

        uint group_size = this->dim_to_size(0, dim);
        uint work_size = this->dim_to_size(dim + 1);
        uint x1_offset = 0, x2_offset = 0, y_offset = 0;
        for (uint i = 0; i < group_size; i++)
        {
            for (uint j = 0; j < this->dim()[dim]; ++j)
            {
                for (uint k = 0; k < work_size; k++)
                {
                    x1_grad[x1_offset++] += y_grad[y_offset++];
                }
            }

            for (uint j = 0; j < x2.dim()[dim]; ++j)
            {
                for (uint k = 0; k < work_size; k++)
                {
                    x2_grad[x2_offset++] += y_grad[y_offset++];
                }
            }
        }

        return x1_grad;
    }


#pragma endregion
#pragma region privates
private:
    Vector<T> &vector()
    {
        _to_host();
        return *_vector;
    }

    const Vector<T> &vector() const
    {
        _to_host();
        return *_vector;
    }

    const Ptr<Vector<T>> &vector_ptr() const
    {
        _to_host();
        return _vector;
    }

    Ptr<Vector<T>> &vector_ptr()
    {
        _to_host();
        return _vector;
    }

    const T &operator[](uint i) const
    {
        return vector()[i];
    }

    T &operator[](uint i)
    {
        return vector()[i];
    }
#pragma endregion
#pragma region binary map & reduce internals
private:
    /****************************************************************************************************************/
    /*  below are major tensor math ops
    add is to output results to y tensor, if add_to is enabled, will keep original value in y; return value is y's ref';
    add_ is to output results to self, may change shape() for some ops, return value is *this's ref, add_to is false in this case;
    above applies to all ops, except explicit annoucements;
    */

    // 1) below are binary ops: tensor op tensor => tensor
    // TODO: define input output tensor shapes

    // generic func for bin_map & bin_reduce
    // bin_op: x1(this) op x2 => y
    // if y is empty, will init shape first, else we assume the y.dim() is already well set.
    // mem-note: tmp tensor for y if y == this && dim changed or reduce
    // input assert: x1 & x2 both is non empty
    // for map: [a,b, c, cross, d, e], a,b,c are first_match_dims,
    // cross is one dim from x1 & x2
    // for map: d, e are last_work_dims
    // for reduce: just 0 dim here, [a, b, c, cross]
    // first split x1(this) and x2 to groups by first_matching_dims, and each group in x1 and x2 will match one by one by id,
    // and then last_work_dims will compose one unit, elems in unit will do map or reduce ops from x1 to x2
    // if there are additional dims in x1 and x2, they will cross join and generate each row
    // first_match_dims: for x1(this) and x2: the first first_match_dims dims will be used as keys, match one by one from x1 and x2, 0 means no need to do this
    // last_work_dims: the last last_work_dims will be used as unit dim for map/reduce ops, 0 means use each elem in x2 as unit, -1 means use full x2 as unit
    // remaining dims besides first_match_dims and last_work_dims will be cross to generate rows for each
    // not-supported yet: x1_key_dims is x1's starting dims before first_match_dims, and x1_id_start/len are used to define partial execution on x1, this
    // is only used by rnn's dot now
    // TODO: not supported yet: uint  = 0, uint x1_id_start = 0, uint x1_id_len = 1) const
    // x1_id: we use x1's first dim as index, and only execute on the matched id
    // yput shape: first dim in x1 will NOT be used in yput shape(), we could treat we use first dim == x1_id to retrieve a sub tensor
    void _bin_preprocess(bool map_or_reduce, const TensorD<T> &x2, const uint first_match_dims, int &last_work_dims, uint &group_count, uint &last_work_size, uint &x1_row, uint &x2_row, Vector<uint> &y_dim) const
    {
        // input verification
        assert(this->shape() > 0 && x2.shape() > 0);
        if (last_work_dims < 0)
        {
            last_work_dims = x2.shape() - first_match_dims;
            assert(last_work_dims >= 0);
        }

        assert(dim().match_bottom(x2.dim(), first_match_dims, false));
        assert(dim().match_bottom(x2.dim(), last_work_dims, true));
        assert(first_match_dims + last_work_dims <= this->shape());
        assert(first_match_dims + last_work_dims <= x2.shape());

        // prepare for group, x1/x2 row, last_work_size
        group_count = dim_to_size(0, first_match_dims);
        last_work_size = dim_to_size(0, last_work_dims, false);
        x1_row = last_work_size == 0 or group_count == 0 ? 0 : this->size() / group_count / last_work_size;
        x2_row = last_work_size == 0 or group_count == 0 ? 0 : x2.size() / group_count / last_work_size;

        // generate y_dim
        if (first_match_dims > 0)
            y_dim.append(dim(), 0, first_match_dims);

        y_dim.append(this->dim(), first_match_dims, this->shape() - first_match_dims - last_work_dims);
        y_dim.append(x2.dim(), first_match_dims, x2.shape() - first_match_dims - last_work_dims);

        if (map_or_reduce)
            y_dim.append(dim(), this->shape() - last_work_dims, last_work_dims); // if it's reduce, only one elem instead of last_work_dims

        if (y_dim.size() == 0)
        {
            y_dim.push_back(1);
        }
    }

    TensorD<T> &_bin_func(const std::function<T(T, T)> &map_func, const std::function<T(T, T, uint)> &reduce_func,
                          bool map_or_reduce, const TensorD<T> &x2, TensorD<T> &y,
                          uint first_match_dims, int last_work_dims) const
    {
        // assert(this != &y && &x2 != &y); this is risky
        //  preprocess
        uint group_count, last_work_size, x1_row, x2_row;
        Vector<uint> y_dim;
        this->_bin_preprocess(map_or_reduce, x2, first_match_dims, last_work_dims, group_count, last_work_size, x1_row, x2_row, y_dim);

        // to support op_
        TensorD<T> *y_ptr = &y;
        bool tmp_y = false;
        if (y_ptr == this && (!map_or_reduce || x2_row > 1)) // results will write to x1, if x2_row == 1, still could use x1 directly
        {
            y_ptr = new TensorD<T>();
            tmp_y = true;
        }

        y_ptr->reset(y_dim); // previous data is cleaned up

        // final processing is 4 layer loops:
        // layer-1: group matching, layer-2: x1_row, layer-3: x2_row, layer-4: last_work_size vector map/reduce
        uint x1_start = 0, x2_start = 0, y_start = 0;
        for (uint g = 0; g < group_count; ++g)
        {
            for (; x1_start < this->size() / group_count * (g + 1); x1_start += last_work_size)
            {
                for (x2_start = x2.size() / group_count * g; x2_start < x2.size() / group_count * (g + 1); x2_start += last_work_size)
                {
                    if (map_or_reduce)
                    {
                        this->vector().map(map_func, x2.vector(), y_ptr->vector(), x1_start, x2_start, y_start, last_work_size /*, add_to*/);
                        y_start += last_work_size;
                    }
                    else
                    {
                        T res = this->vector().reduce(map_func, reduce_func, x2.vector(), x1_start, x2_start, last_work_size);
                        // if (add_to)
                        //     (*y_ptr)[y_start] += res;
                        // else
                        y_ptr->vector()[y_start] = res;
                        y_start++;
                    }
                }
            }
        }

        if (tmp_y) // copy back to this, violate const actually
        {
            y.weak_copy(*y_ptr);
            delete y_ptr;
        }

        return y;
    }

    // bin_map
    inline TensorD<T> &_map(const std::function<T(T, T)> &func, const TensorD<T> &x2, TensorD<T> &y,
                            uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return _bin_func(func, Math<T>::empty_reduce, true, x2, y, first_match_dims, last_work_dims);
    }

    // bin_reduce
    inline TensorD<T> &_reduce(const std::function<T(T, T)> &map_func, const std::function<T(T, T, uint)> &reduce_func,
                               const TensorD<T> &x2, TensorD<T> &y,
                               uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return _bin_func(map_func, reduce_func, false, x2, y, first_match_dims, last_work_dims);
    }

    inline TensorD<T> &_sum_func(const std::function<T(T, T)> &map_func, const TensorD<T> &x2, TensorD<T> &y,
                                 uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return _reduce(map_func, Math<T>::sum_reduce, x2, y, first_match_dims, last_work_dims);
    }

    // reduce(reduce(reduce(map(x11, x21), map(x12, x22)), map(x13, x23)), map(x14, x24))
    // Note: for reduce_grad, only support sum_func so far, i.e., map(x11, x21) + map(x12, x22) + ...
    TensorD<T> &_bin_func_grad(const std::function<void(T, T, T, T &, T &)> &map_grad_func,
                               // const std::function<T(T, T)> &reduce_grad_func,
                               bool map_or_reduce, const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                               TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                               uint first_match_dims = 0, int last_work_dims = -1) const
    {
        // preprocess
        assert(this != &y && this != &y_grad && this != &x2_grad &&
               this != &x1_grad && &y != &y_grad && &y != &x1_grad && &y != &x2 && &y != &x2_grad && &y_grad != &x1_grad &&
               &y_grad != &x2 && &y_grad != &x2_grad && &x1_grad != &x2 && &x2 != &x2_grad);
        uint group_count, last_work_size, x1_row, x2_row;
        Vector<uint> y_dim; // not used
        this->_bin_preprocess(map_or_reduce, x2, first_match_dims, last_work_dims, group_count, last_work_size, x1_row, x2_row, y_dim);
        // TODO: didn't check y & y_grad dims, since most probably no need

        // prepare output dims, don't clear original data since it may already have data
        if (x1_grad.size() == 0)
        {
            x1_grad.reset(dim(), TensorInit_Types::Zero);
        }

        if (x2_grad.size() == 0 && enable_x2_grad)
        {
            x2_grad.reset(x2.dim(), TensorInit_Types::Zero);
        }

        uint x1_group_size = this->size() / group_count;
        uint x2_group_size = x2.size() / group_count;
        uint y_group_size = y.size() / group_count;

        uint x1_start = 0, x2_start = 0, y_start = 0;
        for (uint g = 0; g < group_count; ++g)
        {
            for (uint x1_row_id = 0; x1_row_id < x1_row; ++x1_row_id)
            {
                for (uint x2_row_id = 0; x2_row_id < x2_row; ++x2_row_id)
                {
                    // TODO: use Vector<T>::bin_map => needs bin_map to output 2 vectors
                    for (uint w = 0; w < last_work_size; ++w)
                    {
                        T xe1_grad, xe2_grad;
                        uint x1_id = g * x1_group_size + x1_row_id * last_work_size + w;
                        uint x2_id = g * x2_group_size + x2_row_id * last_work_size + w;

                        uint y_id;
                        if (map_or_reduce)
                        {
                            y_id = g * y_group_size + x1_row_id * x2_row * last_work_size + x2_row_id * last_work_size + w;
                        }
                        else
                        {
                            y_id = g * y_group_size + x1_row_id * x2_row + x2_row_id;
                        }

                        T ye = y[y_id];
                        map_grad_func((*this)[x1_id], x2[x2_id], ye, xe1_grad, xe2_grad);

                        T ye_grad = y_grad[y_id];

                        // note: use += to accumulate gradients
                        x1_grad[x1_id] += xe1_grad * ye_grad;
                        if (enable_x2_grad)
                            x2_grad[x2_id] += xe2_grad * ye_grad;
                    }
                }
            }
        }

        return x1_grad;
    }

    inline TensorD<T> &_map_grad(const std::function<void(T, T, T, T &, T &)> &map_grad_func,
                                 const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                                 TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                                 uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return _bin_func_grad(map_grad_func, /*Math<T>::_empty_reduce, */ true, x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // this is reduce's sum func grad
    inline TensorD<T> &_sum_func_grad(const std::function<void(T, T, T, T &, T &)> &map_grad_func,
                                      const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                                      TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                                      uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return _bin_func_grad(
            map_grad_func,
            //[x2, y, y_grad, &x1_grad, &x2_grad, enable_x2_grad, first_match_dims, last_work_dims](T v, T res) -> T
            //[] { T, T } -> T { return 1; },
            false, x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

#define BIN_REDUCE(name, map_func) \
    TensorD<T> &name(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const { return _sum_func(map_func, x2, y, first_match_dims, last_work_dims); };
#define BIN_REDUCE_GRAD(grad_name, grad_func)                                                                                      \
    TensorD<T> &grad_name(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,                                     \
                          TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,                                   \
                          uint first_match_dims = 0, int last_work_dims = -1) const                                                \
    {                                                                                                                              \
        return this->_sum_func_grad(grad_func, x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims); \
    }

#pragma endregion
#pragma region unary map & reduce internals
private:
    // _uni_op is a special case of _bin_op, there is only one tensor x1(this), and there is no first_match_dims
    // constraint: if it's reduce or x1_id >= 0, can't yput to self
    // constraint: if it's yput to self, add_to is set to be false, to avoid mis-use
    // mem-note: no temp tensor generated
    // if last_work_dims == -1 && x1_id >= 0: we need to ensure actual last_work_dims skip first dim

    // removed x1_id supports to make interface simple
    void _un_preprocess(int last_work_dims, bool map_or_reduce, bool is_grad,
                        uint &last_size, Vector<uint> &y_dim) const
    {
        assert(this->shape() > 0);

        if (last_work_dims < 0)
        {
            last_work_dims = shape();
        }
        else if (last_work_dims > 0)
        {
            assert(last_work_dims <= this->shape());
        }

        last_size = dim_to_size(0, last_work_dims, false);

        // calc y_dim
        if (map_or_reduce || is_grad)
        {
            y_dim.copy(dim());
        }
        else
        {
            y_dim.copy(dim(), 0, this->shape() - last_work_dims);

            if (y_dim.size() == 0) // that means for tensor will reduce to one float
            {
                y_dim.push_back(1);
            }
        }
    }

    // this is low level listwise map
    TensorD<T> &_map(const std::function<void(const Vector<T> &, Vector<T> &, uint, uint, int)> &map_func, TensorD<T> &y,
                     int last_work_dims = -1) const
    {
        uint last_size;
        Vector<uint> y_dim;
        _un_preprocess(last_work_dims, true, false, last_size, y_dim);

        y.reset(y_dim);

        for (uint start = 0; start < this->size(); start += last_size)
        {
            map_func(this->vector(), y.vector(), start, start, (int)last_size);
        }

        return y;
    }

    // constraint: this != &x1_grad, i.e., can't yput to self
    // for map, if not consider x1_id, x1(this)'s dim == y's dim == y_grad's dim
    TensorD<T> &_map_grad(const std::function<void(const Vector<T> &, const Vector<T> &, const Vector<T> &, Vector<T> &, uint, uint, uint, int)> &map_grad_func,
                          const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                          int last_work_dims = -1) const
    {
        // first, let's use default parameters for last_work_dims, add_to, x1_id
        // and then let's do map_func first, y = f(x1); x1_grad = f'(x1) * y_grad,
        // f'() is the Jaccobean matrix, for map(), it's just a linear vector

        assert(/*this != &y &&*/ this != &y_grad && this != &x1_grad && &y != &y_grad && &y != &x1_grad && &y_grad != &x1_grad);
        assert(y.dim().equals_to(y_grad.dim()));
        assert(this != &x1_grad); // can't assign grad to self

        uint last_size;
        Vector<uint> x1_grad_dim;
        _un_preprocess(last_work_dims, true, true, last_size, x1_grad_dim);

        // note: used to accumulate gradients
        if (x1_grad.size() == 0)
        {
            x1_grad.reset(x1_grad_dim, TensorInit_Types::Zero);
        }

        for (uint start = 0; start < this->size(); start += last_size)
        {
            map_grad_func(this->vector(), y.vector(), y_grad.vector(), x1_grad.vector(), start, start, start, last_size);
        }

        return x1_grad;
    }

    // this is high level pointwise map
    // TODO: we no need to use low level pointwise map to speed up
    TensorD<T> &_map(const std::function<T(T)> &map_func, TensorD<T> &y, int last_work_dims = -1) const
    {
        return _map([map_func](const Vector<T> &xv1, Vector<T> &y, uint xe1_start, uint y_start, int last_size) -> void
                    { xv1.map(map_func, y, xe1_start, y_start, last_size); },
                    y, last_work_dims);
    }

    TensorD<T> &_map_grad(const std::function<T(T, T)> &map_grad_func,
                          const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                          int last_work_dims = -1) const
    {
        auto map_grad_func1 = [map_grad_func](const Vector<T> &x, const Vector<T> &y, const Vector<T> &y_grad, Vector<T> &x_grad,
                                              uint x_start, uint y_start, uint x_grad_start, uint len) -> void
        {
            for (uint i = 0; i < len; ++i)
            {
                x_grad[x_grad_start + i] += map_grad_func(x[x_start + i], y[y_start + i]) * y_grad[y_start + i];
            }
        };

        return _map_grad(map_grad_func1, y, y_grad, x1_grad, last_work_dims);
    }

private:
    // this is listwise reduce func, this is low-level one
    TensorD<T> &_reduce(const std::function<T(const Vector<T> &, uint, int)> &reduce_func,
                        TensorD<T> &y, int last_work_dims = -1) const
    {
        assert(this != &y);

        uint last_size;
        Vector<uint> y_dim;
        _un_preprocess(last_work_dims, false, false, last_size, y_dim);
        y.reset(y_dim);

        uint y_start = 0;
        for (uint start = 0; start < this->size(); start += last_size)
        {
            T res = reduce_func(this->vector(), start, last_size);
            y[y_start] = res;
            y_start++;
        }

        return y;
    }

    // void reduce_grad_func(const Tensor<T>& x1, uint x1_start, int len, const T& y, Tensor<T>& x1_grad, uint x1_grad_start);
    TensorD<T> &_reduce_grad(const std::function<void(const Vector<T> &, uint, int, T, T, Vector<T> &, uint)> &reduce_grad_func,
                             const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                             int last_work_dims = -1) const
    {
        // first, let's use default parameters for last_work_dims, add_to, x1_id
        // and then let's do map_func first, y = f(x1); x1_grad = f'(x1) * y_grad,
        // f'() is the Jaccobean matrix, for map(), it's just a linear vector

        assert(this != &y && this != &y_grad && this != &x1_grad && &y != &y_grad && &y != &x1_grad && &y_grad != &x1_grad);
        assert(y.dim().equals_to(y_grad.dim()));
        assert(this != &x1_grad); // can't assign grad to self

        uint last_size;
        Vector<uint> x1_grad_dim;
        _un_preprocess(last_work_dims, false, true, last_size, x1_grad_dim);

        // note: used to accumulate gradients
        if (x1_grad.size() == 0)
        {
            x1_grad.reset(x1_grad_dim, TensorInit_Types::Zero);
        }

        uint y_start = 0, x1_grad_start = 0;
        for (uint start = 0; start < this->size(); start += last_size)
        {
            // for reduce, f1 = f(x1[1]), f2 = f(x1[2]), ...,  gn = g(gn-1, fn)
            // dgn_dfn = g'(fn), e.g., for sum_reduce, it's 1, for product_reduce, it's product of all other elems
            // gn = g(gn-1, fn) = g(g(gn-2, fn-1), fn) = g(g...g(gi-1, fi)), dgn_dfi = g'(gn-1) * g'(gn-2) * ... * g'(fi)
            // and then reduce grad func should be: reduce_grad_func(const Tensor<T>& x1, const T& y, Tensor<T>& x1_grad);
            reduce_grad_func(this->vector(), start, last_size, y[y_start], y_grad[y_start], x1_grad.vector(), x1_grad_start);
            // x1_grad._vector->linear_(x1_grad_start, last_size, y_grad[y_start]); // mul up level grad, TODO merge with above loop
            y_start++;
            x1_grad_start += last_size;
        }

        return x1_grad;
    }

    // this is pointwise sequential reduce func, special case of above
    // for y = reduce(agg, x), we just need to implement dy_dagg, and dy_dx, and then we need each time step's
    // x and agg and also y sometimes
    NOGRAD inline TensorD<T> &_reduce(const std::function<T(T)> &map_func, const std::function<T(T, T)> &reduce_func,
                                      TensorD<T> &y, int last_work_dims = -1) const
    {
        return _reduce([this, map_func, reduce_func](const Vector<T> &v, uint xe1_start, int len) -> T
                       { return this->vector().reduce(map_func, reduce_func, xe1_start, len); },
                       y, last_work_dims);
    }

    inline TensorD<T> &_sum_func(const std::function<T(T)> &map_func, TensorD<T> &y,
                                 int last_work_dims = -1) const
    {
        return _reduce(map_func, Math<T>::sum_reduce0, y, last_work_dims);
    }

    inline TensorD<T> &_sum_func_grad(const std::function<T(T)> &map_grad_func, const TensorD<T> &y, const TensorD<T> &y_grad,
                                      TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        auto reduce_grad_func = [map_grad_func](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            for (uint i = 0; i < len; ++i)
            {
                T dy_dx = map_grad_func(x1[x1_start + i]);
                x1_grad[x1_grad_start + i] += dy_dx * y_grad;
            }
        };

        return _reduce_grad(reduce_grad_func, y, y_grad, x1_grad, last_work_dims);
    }

// note: __VA_ARGS__ is not used
#define MAP(name, map_func, ...)                                   \
    TensorD<T> &name(TensorD<T> &y, int last_work_dims = -1) const \
    {                                                              \
        return _map(map_func, y, last_work_dims);                  \
    }

#define MAP_GRAD(name, map_grad_func, ...)                                               \
    TensorD<T> &name(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, \
                     int last_work_dims = -1) const                                      \
    {                                                                                    \
        return _map_grad(map_grad_func, y, y_grad, x1_grad, last_work_dims);             \
    }

#pragma endregion
#pragma region cuda
private:
    bool _run_cuda(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims, int last_work_dims, std::function<void(const float *, const float *, float *, uint, uint, uint, uint)> func, bool is_reduce = false) const
    {
        if (CudaCtx<T>::runnable())
        {
            auto sizes = _to_device(x2, y, first_match_dims, last_work_dims, is_reduce);
            if (sizes[0] == 1)
            {
                func(this->_device_vector, x2._device_vector, y._device_vector, sizes[1], sizes[2], sizes[3], sizes[4]);
                return true;
            }
        }

        return false;
    }

    bool _run_cuda_grad(const TensorD<T> &x2, const TensorD<T> &y_grad, TensorD<T> &x1_grad, TensorD<T> &x2_grad, uint first_match_dims, int last_work_dims, std::function<void(const float *, const float *, const float *, float *, float *, uint, uint, uint, uint)> func, bool is_reduce = false) const
    {
        if (CudaCtx<T>::runnable())
        {
            auto sizes = _to_device_grad(x2, y_grad, x1_grad, x2_grad, first_match_dims, last_work_dims, is_reduce);
            if (sizes[0] == 1)
            {
                func(this->_device_vector, x2._device_vector, y_grad._device_vector, x1_grad._device_vector, x2_grad._device_vector,
                     sizes[1], sizes[2], sizes[3], sizes[4]);
                return true;
            }
        }

        return false;
    }

    bool _run_cuda(TensorD<T> &y, int last_work_dims, std::function<void(const float *, float *, uint, uint)> func, bool is_reduce = false) const
    {
        if (CudaCtx<T>::runnable())
        {
            auto sizes = _to_device(y, last_work_dims, is_reduce);
            if (sizes[0] == 1)
            {
                func(this->_device_vector, y._device_vector, sizes[1], sizes[2]);
                return true;
            }
        }

        return false;
    }

    bool _run_cuda_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims, 
    std::function<void(const float *, const float *, const float *, float *, uint, uint)> func, bool is_reduce = false) const
    {
        if (CudaCtx<T>::runnable())
        {
            auto sizes = _to_device_grad(y, y_grad, x1_grad, last_work_dims, is_reduce);
            if (sizes[0] == 1)
            {
                func(this->_device_vector, y._device_vector, y_grad._device_vector, x1_grad._device_vector, sizes[1], sizes[2]);
                return true;
            }
        }

        return false;
    }

    bool _run_cuda_grad(const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims, 
    std::function<void(const float *, const float *, float *, uint, uint)> func, bool is_reduce = false) const
    {
        if (CudaCtx<T>::runnable())
        {
            auto sizes = _to_device_grad(y_grad, x1_grad, last_work_dims, is_reduce);
            if (sizes[0] == 1)
            {
                func(this->_device_vector, y_grad._device_vector, x1_grad._device_vector, sizes[1], sizes[2]);
                return true;
            }
        }

        return false;
    }

    void _to_host() const
    {
        if (_vector == nullptr) // must be the data stored in cuda, i.e., _device_vector
        {
            _vector = std::make_shared<Vector<T>>(this->size());
            if (_device_vector != nullptr)
            {
                CudaCtx<T>::to_host(_vector->ptr(), _device_vector, this->size());
                CudaCtx<T>::free_device(_device_vector);
            }

            _device_vector = nullptr;
        }
    }

    Vector<uint> _to_device(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims, int last_work_dims, bool is_reduce = false) const
    {
        // x1, x2 to device
        if (!_to_device(*this) || !_to_device(x2))
        {
            return {0};
        }

        // allocate device memory for y
        auto sizes = this->_calc_size(x2, first_match_dims, last_work_dims, is_reduce);

        if (y._device_vector != nullptr)
        {
            CudaCtx<T>::free_device(y._device_vector);
        }

        y._device_vector = CudaCtx<T>::alloc_device(sizes[4]);
        if (y._device_vector == nullptr)
        {
            return {0};
        }

        y._vector = nullptr;

        // set y's dim
        if (last_work_dims < 0)
            last_work_dims = x2.shape() - first_match_dims;
        assert(this != &y);
        y._dim.clear();
        y._dim.append(this->dim().subset(0, first_match_dims));
        y._dim.append(this->dim().subset(first_match_dims, this->dim().size() - first_match_dims - last_work_dims));
        y._dim.append(x2.dim().subset(first_match_dims, x2.dim().size() - first_match_dims - last_work_dims));
        if (!is_reduce)
        {
            y._dim.append(this->dim().subset(this->dim().size() - last_work_dims, last_work_dims));
        }

        if (y._dim.size() == 0)
        {
            y._dim.push_back(1);
        }

        return {1, sizes[0], sizes[1], sizes[2], sizes[3]};
    }

    Vector<uint> _to_device_grad(const TensorD<T> &x2, const TensorD<T> &y_grad, TensorD<T> &x1_grad, TensorD<T> &x2_grad, uint first_match_dims, int last_work_dims, bool is_reduce = false) const
    {
        // x1, x2, y_grad to device
        if (!_to_device(*this) || !_to_device(x2) || !_to_device(y_grad))
        {
            return {0};
        }

        // allocate device memory for x1_grad and x2_grad, and set dims
        if (!x1_grad.dim().equals_to(this->dim()))
        {
            x1_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        if (!_to_device(x1_grad))
        {
            return {0};
        }

        if (!x2_grad.dim().equals_to(x2.dim()))
        {
            x2_grad.reset(x2.dim(), TensorInit_Types::Zero);
        }

        if (!_to_device(x2_grad))
        {
            return {0};
        }
        
        auto sizes = this->_calc_size(x2, first_match_dims, last_work_dims, is_reduce);
        return {1, sizes[0], sizes[1], sizes[2], sizes[3]};
    }

    Vector<uint> _to_device(TensorD<T> &y, int last_work_dims, bool is_reduce = false) const
    {
        // x1 to device
        int cols = 0;
        if (!_to_device(*this))
        {
            return {0};
        }

        // allocate device memory for y
        auto sizes = this->_calc_size(last_work_dims, is_reduce);
        if (y._device_vector == nullptr)
        {
            CudaCtx<T>::free_device(y._device_vector);
        }

        y._device_vector = CudaCtx<T>::alloc_device(sizes[2]);
        if (y._device_vector == nullptr)
        {
            return {0};
        }

        y._vector = nullptr;

        // set y's dim
        if (last_work_dims < 0)
            last_work_dims = this->shape();
        assert(this != &y);
        if (!is_reduce)
        {
            y._dim = this->dim();
        }
        else
        {
            y._dim.clear();
            y._dim.append(this->dim().subset(0, this->dim().size() - last_work_dims));
            if (y._dim.size() == 0)
            {
                y._dim.push_back(1);
            }
        }

        return {1, sizes[0], sizes[1]};
    }

    Vector<uint> _to_device_grad(const TensorD<T>& y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims, bool is_reduce = false) const
    {
        if (!_to_device(y))
        {
            return {0};
        }

        return _to_device_grad(y_grad, x1_grad, last_work_dims, is_reduce);
    }

    Vector<uint> _to_device_grad(const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims, bool is_reduce = false) const
    {
        if (!_to_device(*this) || !_to_device(y_grad))
        {
            return {0};
        }

        if (!x1_grad.dim().equals_to(this->dim()))
        {
            x1_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        if (!_to_device(x1_grad))
        {
            return {0};
        }

        auto sizes = this->_calc_size(last_work_dims, is_reduce);
        return {1, sizes[0], sizes[1]};
    }

    static bool _to_device(const TensorD<T> &x)
    {
        if (x._device_vector == nullptr)
        {
            if (x._vector != nullptr)
            {
                x._device_vector = CudaCtx<T>::to_device(x._vector->ptr(), x.size());
                x._vector = nullptr;
                return true;
            }
            else
            {
                x._vector = std::make_shared<Vector<T>>(x.size());
                return false;
            }
        }

        return true;
    }

    Vector<uint> _calc_size(const TensorD<T> &x2, uint first_match_dims, int last_work_dims, bool is_reduce = false) const
    {
        uint y_size = 0;

        assert(first_match_dims >= 0 && first_match_dims <= this->dim().size());
        assert(first_match_dims >= 0 && first_match_dims <= x2.dim().size());
        uint depths = this->dim_to_size(0, first_match_dims);
        y_size = depths;

        if (last_work_dims < 0)
            last_work_dims = x2.dim().size() - first_match_dims;
        assert(last_work_dims >= 0 && last_work_dims <= this->dim().size() - first_match_dims);
        assert(last_work_dims >= 0 && last_work_dims <= x2.dim().size() - first_match_dims);
        uint x1_row = this->dim_to_size(first_match_dims, this->dim().size() - first_match_dims - last_work_dims);
        uint x2_row = x2.dim_to_size(first_match_dims, x2.dim().size() - first_match_dims - last_work_dims);
        y_size *= x1_row * x2_row;

        uint cols = this->dim_to_size(0, last_work_dims, false);

        if (!is_reduce)
        {
            y_size *= cols;
        }

        return {depths, x1_row, x2_row, cols, y_size};
    }

    Vector<uint> _calc_size(int last_work_dims, bool is_reduce = false) const
    {
        if (last_work_dims < 0)
            last_work_dims = this->dim().size();

        uint rows = this->dim_to_size(0, this->dim().size() - last_work_dims);
        uint cols = this->dim_to_size(0, last_work_dims, false);
        return {rows, cols, is_reduce ? rows : this->size()};
    }
#pragma endregion
};
