#pragma once

#include "vector.h"

#pragma region comments
/*
Tensor is an n-dim array, data is stored in an continuous array that low dimensions are stored together first, same as C++ array
below are utility functions, could be implemented on CPU and GPU
note: separate data & ops? => no need
TODO: random ops
TODO: for reduce, now it's a sequential ops, we could change it to parrallel
TODO: support full SQL ops in tensor: where, group-by, etc.
TODO: multi-threading, multi-processing;
TODO: tensor with other data types? for now double, should we support float, half-float, long, int, short, byte, bool;
TODO: use view() to generate logical tensors?
TODO: Tensor should not inherit public functions directly unless _dim.size() == 1
*/

/*
    core ops: swap, move_forward, _bin_op, _uni_op, TODO: tri_op


    supported creation ops:
        ctor: default, dim list, dim list + init_type, copy ctor
        init, copy, clear
        TODO: save, load

    supported retrieval ops:
        dim, item, size, shape, is_empty, dim_to_size, size_to_dim

    supported update ops:
        swap/move_forward
        TODO(from pytorch): flatten, reshape(light_reshape), cat, chunk, gather, index_select, split, transpose

    supported bin ops include: _bin_op,
        map: bin_map, add/add_, mul/mul_,
            TODO: tri_map
        reduce: bin_reduce, bin_sum, bin_product, dot, mse, cross_entropy,
            TODO: cosine_distance, euclidean
        common interface as below:
        Tensor<T> &dot(const Tensor<T> &x2, Tensor<T> &y, uint first_match_dims = 0, int last_work_dims = -1,
                    bool add_to = false, int x1_id = -1) const;
    supported uni ops include: _uni_op,
        map: uni_map, linear/linear_, softmax/softmax_, activation/activation_, sqrt, pow,
            TODO: log, exp, sin/cos, asin/acos, sort, topk, clamp,
            bool TODOs: eq, ge/gt, le/lt,
            probability TODOs: histc(histogram), erf;
            TODO: spectral ops, bit ops, matrix ops, BLAS/LAPACK
        reduce: uni_reduce, uni_sum, uni_product, sum, product, avg, var, max, min, TODO: norm, median,
        common interface as below:
        Tensor<T> &sqrt(Tensor<T> &y, int last_work_dims = -1, bool add_to = false, int x1_id = -1) const;
*/
#pragma endregion

template <class T>
class TensorD;
typedef Ptr<TensorD<double>> TensorDP;
typedef Array<TensorDP> TensorDPArray;
template <typename T>
using TensorDArray = Array<TensorD<T>>;

template <class T>
class TensorD
{
    friend class TestTensor;
    friend class TestTensorNode;
    friend class TestFc;
    friend class TestConv;
    friend class TestPooling;
    friend class TestTransformer;
    friend class TestRnn;
    friend class TestNorm;
    friend class TestFunctorGraph;
    friend class TestOptimizers;
    friend class TestDataLoaders;
    friend class Gbdt;

#pragma region privates
private:
    // changed _vector from inheritance to one member pointer, so that it could support virtual tensors that redirect memory
    // to other tensors, users need to ensure memory is there
    Ptr<Vector<T>> _vector;
    Ptr<Vector<uint>> _dim;
    // uint _size; cache for _dim.product(), no need such small optimization
    // uint _start; used for virtual tensor, disabled now

    Vector<T> &vector()
    {
        return *_vector;
    }

    const Vector<T> &vector() const
    {
        return *_vector;
    }

    const T &operator[](uint i) const
    {
        return (*_vector)[i];
    }

    T &operator[](uint i)
    {
        return (*_vector)[i];
    }
#pragma endregion
public:
#pragma region creates
    /***********************************************************************************************************/
    // below are tensor creation functions

    // default tensor is an empty vector
    TensorD()
    {
        this->_vector = std::make_shared<Vector<T>>();
        this->_dim = std::make_shared<Vector<uint>>();
    }

    explicit TensorD(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None) : TensorD()
    {
        this->reset(dim, t);
    }

    TensorD(const TensorD<T> &x2) : TensorD()
    {
        this->weak_copy(x2);
    }

    TensorD<T> &reset(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None)
    {
        assert(dim.all_pos());
        if (this->size() != dim.product()) // if total size is the same, no need to re-alloc memory, but the data is random
        {
            this->_vector = std::make_shared<Vector<T>>();
            this->_vector->reserve(dim.product());
        }

        if (!this->_dim->equals_to(dim)) // over-optimize, equals_to also spend time
        {
            this->_dim->copy(dim);
        }

        _vector->init(t);
        return *this;
    }

    // perf: if copy from same shape tensor, no need to clear() and init();
    TensorD<T> &deep_copy(const TensorD<T> &x2)
    {
        this->reset(x2.dim());
        this->_vector->set(0, *x2._vector);
        return *this;
    }

    TensorD<T> &weak_copy(const TensorD<T> &x2)
    {
        this->_dim = x2.dim_ptr();
        this->_vector = x2._vector;
        return *this;
    }

    TensorD<T> &deep_dim_copy(const TensorD<T> &x2)
    {
        this->_dim = std::make_shared<Vector<uint>>();
        this->_dim->copy(x2.dim());
        this->_vector = x2._vector;
        return *this;
    }

#ifdef DISABLED
    // not used now
    // note: not copy data, readonly sub tensor, this is virtual tensor concept
    const TensorD<T> sub_tensor(uint x1_id) const
    {
        assert(_dim.size() > 0);
        assert(x1_id < _dim[0]);
        TensorD<T> t;
        t._dim = _dim.subset(1);
        t._size = _size / _dim[0];
        t._vector = _vector;
        t._start = t._size * x1_id;
    }
#endif

    OVERRIDE void clear()
    {
        // this->_vector->clear(); after all reference to _vector is removed, dctor will be called automatically
        //this->_vector = nullptr;
        this->_vector->clear();
        //this->_dim = nullptr;
        this->_dim->clear();
    }

    /*// TODO: serialization
    void load(std::istream &in);
    void save(std::ostream &y) const;*/
#pragma endregion
#pragma region retrievals
    /***************************************************************************************************************/
    // below are tensor retrieval functions, all const
    inline const Vector<uint> &dim() const
    {
        return *_dim;
    }

    inline Vector<uint> &dim()
    {
        return *_dim;
    }

    inline const Ptr<Vector<uint>> dim_ptr() const
    {
        return _dim;
    }

    inline uint size() const
    {
        return dim().product();
    }

    // get first item if tensor is a scalar value
    inline T item() const
    {
        return _vector->front();
    }

    inline uint shape() const
    {
        uint res = dim().size();
        //if (res == 1 && dim()[0] == 1)
        //    res = 0;
        return res;
    }

    inline T first_item() const
    {
        assert(size() > 0);
        return this->vector()[0];
    }

    // not used now
    /*void get_dim_ids(uint index, Vector<uint> &dim_ids) const;
    int get_flat_index(const Vector<uint> &dim_ids) const;
    int get_next_index(const Vector<uint> &dim_ids) const;
    int get_prev_index(const Vector<uint> &dim_ids) const;*/

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
        assert(from < shape());
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

        return size;
    }
#pragma endregion
private:
#pragma region binary_maps
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
        x1_row = this->size() / group_count / last_work_size;
        x2_row = x2.size() / group_count / last_work_size;

        // generate y_dim
        if (first_match_dims > 0)
            y_dim.append(dim(), 0, first_match_dims);

        if (x1_row >= 1)
            y_dim.append(dim(), first_match_dims, this->shape() - first_match_dims - last_work_dims);
        if (x2_row >= 1)
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
        //assert(this != &y && &x2 != &y); this is risky
        // preprocess
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

public: // below are public bin_map ops
    // add op: y[i] [+]= alpha_x1 * x1[i] + alpha_x2 * x2[i] + beta;
    TensorD<T> &add(const TensorD<T> &x2, TensorD<T> &y, T alpha_x1 = 1.0, T alpha_x2 = 1.0, T beta = 0.0,
                    uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return this->_map([alpha_x1, alpha_x2, beta](T xe1, T xe2) -> T
                          { return xe1 * alpha_x1 + xe2 * alpha_x2 + beta; },
                          x2, y, first_match_dims, last_work_dims);
    }

    TensorD<T> &add_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         T alpha_x1 = 1.0, T alpha_x2 = 1.0, T beta = 0.0,
                         uint first_match_dims = 0, int last_work_dims = -1) const
    {
        // x1_grad all values is alpha_x1, x2_grad all values is alpha_x2
        // TODO: most case alpha_x1 & alpha_x2 are 1, perf optimization
        return this->_map_grad([alpha_x1, alpha_x2](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                               {
                            xe1_grad = alpha_x1;
                            xe2_grad = alpha_x2; },
                               x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // mul op: y[i] [+]= alpha * x1[i] * x2[i] + beta; Hamard product
    TensorD<T> &mul(const TensorD<T> &x2, TensorD<T> &y, T alpha = 1.0, T beta = 0.0,
                    uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return this->_map([alpha, beta](T xe1, T xe2) -> T
                          { return xe1 * alpha * xe2 + beta; },
                          x2, y, first_match_dims, last_work_dims);
    }

    TensorD<T> &mul_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         T alpha = 1.0, T beta = 0.0,
                         uint first_match_dims = 0, int last_work_dims = -1) const
    {
        // x1_grad all values is alpha * x2[i], x2_grad all values is alpha * x1[i]
        return this->_map_grad([alpha](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                               {
                            xe1_grad = alpha * xe2;
                            xe2_grad = alpha * xe1; },
                               x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }
#pragma endregion
#pragma region binary_reduces
public: // below are public bin_reduce ops
#define BIN_REDUCE(name, map_func) \
    TensorD<T> &name(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const { return _sum_func(map_func, x2, y, first_match_dims, last_work_dims); };
#define BIN_REDUCE_GRAD(grad_name, grad_func)                                                                                      \
    TensorD<T> &grad_name(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,                                     \
                          TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,                                   \
                          uint first_match_dims = 0, int last_work_dims = -1) const                                                \
    {                                                                                                                              \
        return this->_sum_func_grad(grad_func, x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims); \
    }

    // dot op: y[i] = sum(j, x1[j] * x2[j])
    // if specify, x1_id, first dim of x1 will be skipped in y shape
    // TODO: supports bias, merge to one op
    /*
    used by:
        0) linear/node-linear: x.dot(w, y);
        1) conv: col.dot(k, y, 1, 3);
        2) rnn: hidden.dot(u, y);
        3) transformer: x.dot(wq, q); q.dot(k, weights, 2, 1); weights.dot(v, y, 2, 1);
    */
    BIN_REDUCE(dot, Math<T>::multi_op)
    // x1_grad all values is x2[i], x2_grad all values is x1[i]
    BIN_REDUCE_GRAD(dot_grad, [](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                    {
                        xe1_grad = xe2;
                        xe2_grad = xe1; });

    // mse op: y[i] = avg(j, (x1[j] - x2[j]) ^ 2)
    // used by MseLoss
    TensorD<T> &mse(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return _sum_func([n](T xe1, T xe2) -> T
                         { return (xe1 - xe2) * (xe1 - xe2) / n; },
                         x2, y, first_match_dims, last_work_dims);
    }

    // x1_grad all values is 2(x1[i] - x2[i])/n, x2_grad all values is 2(x2[i] - x1[i])/n
    // y.add(t, y_grad, 0, -1, false, 2.0 / n, -2.0 / n, 0.0);
    TensorD<T> &mse_grad(const TensorD<T> &x2, const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x1_grad, TensorD<T> &x2_grad, bool enable_x2_grad = false,
                         uint first_match_dims = 0, int last_work_dims = 1) const
    {
        uint n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return this->_sum_func_grad([n](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                                    {
                                 xe1_grad = 2 * (xe1 - xe2) / n;
                                 xe2_grad = -1 * xe1_grad; },
                                    x2, y, y_grad, x1_grad, x2_grad, enable_x2_grad, first_match_dims, last_work_dims);
    }

    // ce op: y[i] = -1.0 * sum(j, (x1[j] * log2(x2[j])))
    // used by CrossEntropyLoss
    BIN_REDUCE(ce, [](T xe1, T xe2) -> T
               {
                   assert(xe2 >= 0);
                   xe2 = xe2 == 0 ? xe2 + EPSILON : xe2;
                   return -1.0 * xe1 * std::log2(xe2); });
    // x1_grad all values is log2(x2[j]), x2_grad all values is x1[i] /(x2[i]) / std::log(2)
    BIN_REDUCE_GRAD(ce_grad, [](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                    {
                        assert(xe2 > 0);
                        xe1_grad = -1.0 * std::log2(xe2);
                        xe2_grad = -1.0 * xe1 / xe2 / std::log(2.0); });

    // eu op: y[i] = sqrt(sum(j, ((x1[j] - x2[j])^2))
    TensorD<T> &euclidean(const TensorD<T> &x2, TensorD<T> &y, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        _sum_func([](T xe1, T xe2) -> T
                 { return (xe1 - xe2) * (xe1 - xe2); },
                 x2, y, first_match_dims, last_work_dims);
        y.sqrt(y, last_work_dims);
        return y;
    }

    // grad: dy_dx1 = 0.5 / y * 2 * (x1[j] - x2[j]), dy_dx2 = -dy_dx1
    BIN_REDUCE_GRAD(euclidean_grad, [](T xe1, T xe2, T ye, T &xe1_grad, T &xe2_grad) -> void
                    {
                        assert(xe2 > 0);
                        if (ALMOST_ZERO(ye)){
                            xe1_grad = xe2_grad = 0;
                        }
                        else
                        {
                            xe1_grad = (xe1 - xe2) / ye; xe2_grad = - xe1_grad;
                        }
                    });

#pragma endregion
#pragma region unary_maps
    // 2) below are uni ops: tensor -> tensor, all cloned from above bin ops

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

public:
    // linear: y[i] = x[i] * alpha + beta;
    TensorD<T> &linear(TensorD<T> &y, T alpha = 1.0, T beta = 0.0, int last_work_dims = -1) const
    {
        return _map([alpha, beta](T x) -> T
                    { return x * alpha + beta; },
                    y, last_work_dims);
    }

    TensorD<T> &linear_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                            T alpha = 1.0, T beta = 0.0, int last_work_dims = -1) const
    {
        return _map_grad([alpha](T x, T y) -> T
                         { return alpha; },
                         y, y_grad, x1_grad, last_work_dims);
    }

    MAP(sqrt, Math<T>::sqrt)

    MAP_GRAD(sqrt_grad, [](T ex, T ey) -> T
             { return ey != 0 ? 0.5 / ey : 0; })

    // y = pow(x + bias, n)
    TensorD<T> &pow(TensorD<T> &y, double n, double bias = 0, int last_work_dims = -1) const
    {
        return _map([n, bias](T ex) -> T
                    { return std::pow(ex + bias, n); },
                    y, last_work_dims);
    }

    TensorD<T> &pow_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                         double n, double bias = 0, int last_work_dims = -1) const
    {
        return _map_grad([n, bias](T ex, T ey) -> T
                         { return (ex + bias) != 0 ? n * ey / (ex + bias) : 0; },
                         y, y_grad, x1_grad, last_work_dims);
    }

    /*MAP(
        pow, [n](T x) -> T
        { return std::pow(x, n) },
        double n)

    MAP_GRAD(
        pow_grad, [n](T x, T y) -> T
        { return x != 0 ? n * y / x : 0; },
        double n)*/

    // y[i] = exp(x[i]) / sum(exp(x[i]))
    // TODO: use low level map to implement
    TensorD<T> &softmax(TensorD<T> &y, int last_work_dims = -1) const
    {
        /* this is wrong since we need to have denominator for each last_work_dim
        double denominator = 0;
        this->_map([&denominator](T e) -> T
                   {
                auto exp_e = std::exp(e);
                denominator += exp_e;
                return exp_e; },
                   y, last_work_dims);

        // note: handle denominator == 0 case => no need, since the lambda will not be executed due to len == 0
        return y._map([denominator](T e) -> T
                     { return e / denominator; },
                     y, last_work_dims);*/

        return _map([](const Vector<T> &xv1, Vector<T> &y, uint xv1_start, uint y_start, int last_size) -> void
                    { xv1.softmax(y, xv1_start, y_start, last_size); },
                    y, last_work_dims);
    }

    // not-used: x1(this)
    // d(y/x) = (xdy - ydx) / x^2
    //  dy_j/dx_i = exp(x_j) * -1 * sum(k, exp(x_k))^2 * exp(x_i) = -1 * y_i * y_j if i != j;
    //  dy_i/dx_i = (sum(exp(x_k)) * exp(x_i) - exp(x_i) * exp(x_i)) / sum(exp(x_i))^2 = y_i * (1 - y_i)
    //  dL/dx_i = sum(j, dL/dy_j * dy_j/dx_i) = sum(j, dL/dy_j * dy_j/dx_i or dy_i/dx_i) = sum(j, dL/dy_j * (i == j - y_j) * y_i) =
    //        = y_i * (dL/dy_i - sum(j, dL/dy_j * y_j))
    TensorD<T> &softmax_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, int last_work_dims = -1) const
    {
        // Jacobian Matrix
        auto map_grad_func = [](const Vector<T> &x, const Vector<T> &y, const Vector<T> &y_grad, Vector<T> &x_grad,
                                uint x_start, uint y_start, uint x_grad_start, uint len) -> void
        {
            T sum = y.dot(y_grad, y_start, y_start, len);
            for (uint i = 0; i < len; ++i)
            {
                x_grad[i + x_grad_start] += y[i + y_start] * (y_grad[i + y_start] - sum);
                // y.sum_func([i](T y_j, T y_grad_j, uint j) -> T{ return y_grad_j * ((i == j) - y_j) * y_j; }, y_grad, y_start, y_start, len);
            }
        };

        return this->_map_grad(map_grad_func, y, y_grad, x_grad, last_work_dims);
    }

    TensorD<T> &activation(Activation_Types type, TensorD<T> &y, int last_work_dims = -1) const
    {
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

#pragma endregion
#pragma region unary_reduces
    // below are unary reduce ops
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

    /* disabled, not used
    inline TensorD<T> &product_func(const std::function<T(T)> &map_func, TensorD<T> &y,
                                    int last_work_dims = -1, bool add_to = false) const
    {
        return reduce(map_func, Math<T>::product_reduce, y, last_work_dims, add_to);
    }

    inline TensorD<T> &product_func_grad(const std::function<T(T)> &map_grad_func, const TensorD<T> &y, const TensorD<T> &y_grad,
                                         TensorD<T> &x1_grad, int last_work_dims = -1, bool add_to = false) const
    {
        auto reduce_grad_func = [](const Vector<T> &x1, uint x1_start, int len, T y, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            for (uint i = 0; i < len; ++i)
            {
                if (x1[x1_start + i] == 0)
                {
                    x1_grad[x1_grad_start + i] = 0;
                    LOG_WARN("0 value has no grad in product()");
                }
                else
                {
                    T dy_dx = map_grad_func(x1[x1_start + i]);
                    x1_grad[x1_grad_start + i] = dy_dx * y / x1[x1_start + i];
                }
            }
        };

        return reduce_grad(reduce_grad_func, y, y_grad, x1_grad, last_work_dims, add_to);
    }

    TensorD<T> &product(TensorD<T> &y, int last_work_dims = -1, bool add_to = false) const
    {
        return product(Math<T>::empty_map, y, last_work_dims, add_to);
    }

    TensorD<T> &product_grad(TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims = -1, bool add_to = false) const
    {
        return product_func_grad([](T e) -> T
                                 { return 1; },
                                 y, y_grad, x1_grad, last_work_dims, add_to);
    }
*/
public:
    // TODO: specify a list of dims in the tensor, and reduce each of them by sum, this is a complex function if we don't do optimal
    // time/space complexity
    inline TensorD<T> &sum(TensorD<T> &y, int last_work_dims = -1) const
    {
        return _sum_func(Math<T>::empty_map, y, last_work_dims);
    }

    inline TensorD<T> &sum_grad(const TensorD<T> &y, const TensorD<T> &y_grad,
                                TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        return _sum_func_grad([](T e) -> T
                              { return 1; },
                              y, y_grad, x1_grad, last_work_dims);
    }

    TensorD<T> &avg(TensorD<T> &y, int last_work_dims = -1) const
    {
        T n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        // note: this is not optimal since each unit just needs one /, now each elem needs one /.
        return _sum_func([n](T e) -> T
                         { return e / n; },
                         y, last_work_dims);
    }

    TensorD<T> &avg_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
        T n = dim_to_size(0, last_work_dims, false);
        assert(n > 0);
        return _sum_func_grad([n](T e) -> T
                              { return 1 / n; },
                              y, y_grad, x1_grad, last_work_dims);
    }

    // note: this is var, not stdvar
    TensorD<T> &var(TensorD<T> &y, bool biased = false, int last_work_dims = -1) const
    {
        if (last_work_dims < 0) last_work_dims = this->shape();
        uint first_match_dims = this->shape() - last_work_dims;

        TensorD<T> mean;
        avg(mean, last_work_dims);
        TensorD<T> mean_inf;
        mean.inflate(mean_inf, Vector(this->dim().subset(first_match_dims, last_work_dims))); // make each mean value in last dim to be the size of work_size, all the same value
        // TODO: we do this because current reduce & bin_reduce only support matched work_size, here the work_size is n VS 1

        uint n = dim_to_size(0, last_work_dims, false);
        if (biased) --n;
        assert(n > 0);
        // note: this is not optimal since each unit just needs one /, now each elem needs one /.

        return _sum_func([n](T e, T m) -> T
                         { return (e - m) * (e - m) / n; },
                         mean_inf, y, first_match_dims, last_work_dims);
    }

    TensorD<T> &var_grad(const TensorD<T> &y, const TensorD<T> &y_grad,
                         TensorD<T> &x_grad, bool biased = false, int last_work_dims = -1) const
    {
        auto reduce_grad_func = [](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            uint n = len;
            double m = x1.avg(x1_start, len);
            // dmean_dx = 1/n
            // dvar_dx = 2/n * (e - m) * (1 - 1/n)
            for (uint i = 0; i < len; ++i)
            {
                double dmean_dx = 1.0 / n;
                double dvar_dx = 2.0 / n * (x1[x1_start + i] - m) * (1 - dmean_dx);
                x1_grad[x1_grad_start + i] += dvar_dx * y_grad;
            }
        };

        return _reduce_grad(reduce_grad_func, y, y_grad, x_grad, last_work_dims);
    }

    // sugar: could save 1 time avg()
    /*TensorD<T> &mean_var(TensorD<T> &mean, TensorD<T> &var, bool biased = false, int last_work_dims = -1) const
    {
        avg(mean, last_work_dims);
        uint work_size = dim_to_size(0, last_work_dims, false);
        TensorD<T> mean_inf;
        mean.inflate(mean_inf, work_size); // make each mean value in last dim to be the size of work_size, all the same value
        // TODO: we do this because current reduce & bin_reduce only support matched work_size, here the work_size is n VS 1

        uint n = dim_to_size(0, last_work_dims, false);
        if (!biased)
            --n;
        assert(n > 0);
        // note: this is not optimal since each unit just needs one /, now each elem needs one /.
        return _sum_func([n](T e, T m) -> T
                        { return (e - m) * (e - m) / n; },
                        mean_inf, var, last_work_dims);
    }

    TensorD<T> &mean_var_grad(const TensorD<T> &mean, const TensorD<T> &mean_grad, const TensorD<T> &var, const TensorD<T> &var_grad,
                              TensorD<T> &x_grad, bool biased = false, int last_work_dims = -1) const
    {
        auto reduce_grad_func = [](const Vector<T> &x1, uint x1_start, int len, T y, T y_grad, Vector<T> &x1_grad, uint x1_grad_start) -> void
        {
            if (len < 0)
                len = x1.size() - x1_start;
            assert(x1_start < x1.size());

            uint n = len;
            double m = x1.avg(x1_start, len);
            // dmean_dx = 1/n
            // dvar_dx = 2/n * (e - m) * (1 - 1/n)
            for (uint i = 0; i < len; ++i)
            {
                double dmean_dx = 1.0 / n;
                double dvar_dx = 2.0 / n * (x1[x1_start + i] - m) * (1 - 1.0 / n);
                x1_grad[x1_grad_start + i] += (dmean_dx + dvar_dx) * y_grad;
            }
        };

        return _reduce_grad(reduce_grad_func, y, y_grad, x_grad, last_work_dims);
    }*/

    TensorD<T> &max(TensorD<T> &y, int last_work_dims = -1) const
    {
        return _reduce(
            Math<T>::empty_map, [](T res, T e) -> T
            { return e > res ? e : res; },
            y, last_work_dims);
    }

    TensorD<T> &max_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad, int last_work_dims = -1) const
    {
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
        return _reduce(
            Math<T>::empty_map, [](T res, T e) -> T
            { return e < res ? e : res; },
            y, last_work_dims);
    }

    TensorD<T> &min_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x1_grad,
                         int last_work_dims = -1) const
    {
        return max_grad(y, y_grad, x1_grad, last_work_dims);
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

    /* disabled, swap() is super set of this
     // time-complex: O(n), but will non-conj memory access
     NOGRAD TensorD<T> &swap_adj(TensorD<T> &y, uint first_dim) const
     {
         // at least 2 dims
         assert(shape() > 1 && first_dim < shape() - 1);
         Vector<uint> y_dim(dim());
         uint row_count = y_dim[first_dim] = dim()[first_dim + 1];
         uint col_count = y_dim[first_dim + 1] = dim()[first_dim];
         y.reset(y_dim);

         uint group_count = dim_to_size(0, first_dim);
         uint group_len = size() / group_count;
         uint unit_len = dim_to_size(0, shape() - first_dim - 2, false); // unit move unit, above eg is 5
         uint in_start = 0, y_start = 0;
         for (uint g = 0; g < group_count; ++g)
         {
             for (uint r = 0; r < row_count; ++r)
                 for (uint c = 0; c < col_count; ++c)
                 {
                     in_start = g * group_len + r * unit_len + c * unit_len * row_count; // TODO: perf: change * to +
                     y._vector->set(y_start, this->_vector->get(), in_start, unit_len);
                     y_start += unit_len;
                 }
         }

         return y;
     }
     */

    // swap any two dims in the tensor
    // grad is reverse op
    NOGRAD TensorD<T> &swap(TensorD<T> &y, uint first_dim, uint second_dim) const
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
    NOGRAD TensorD<T> &move_forward(TensorD<T> &y, uint move_from, uint move_len, uint move_to) const
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

    TensorD<T> &im2col(TensorD<T> &y, uint groups = 1, uint kernel_x = 1, uint kernel_y = 1,
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
                                    double v = 0; // default padding value
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
    TensorD<T> &im2col_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad,
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
                                    double v = 0; // default padding value
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

    void divide(TensorDArray<T> &y, uint first_match_dims = 1) const
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
                y[i]._vector->set(0, this->vector(), i * sub_tensor_size, sub_tensor_size);
            }
        }
    }

    void divide_grad(const TensorDArray<T> &y, const TensorDArray<T> &y_grad, TensorD<T> &x_grad, uint first_match_dims = 1) const
    {
        assert(first_match_dims <= shape());
        if (x_grad.dim() != dim())
        {
            x_grad.reset(dim(), TensorInit_Types::Zero);
        }

        for (uint i = 0; i < y_grad.size(); ++i)
        {
            x_grad._vector->set(i * y_grad[i].size(), y_grad[i].vector(), 0, y_grad[i].size());
        }
    }

    static TensorD<T> combine(const TensorDArray<T> &x, const Vector<uint> &first_dims = {})
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

    static void combine_grad(const TensorDArray<T> &x, const TensorD<T> &y, const TensorD<T> &y_grad,
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

        y.dim().copy(dims);

        y._vector = this->_vector;

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

    TensorD<T> &inflate(TensorD<T> &y, const Vector<uint>& dims) const
    {
        if (dims.size() == 0)
        {
            y.deep_copy(*this);
            return y;
        }

        uint inflation_size = dims.product();
        assert(inflation_size > 0);
        Vector<uint> dim = this->dim();
        dim.append(dims);
        y.reset(dim);
        for (uint i = 0; i < this->size(); ++i)
        {
            y.vector().set_each((*this)[i], i * inflation_size, inflation_size);
        }

        return y;
    }

    TensorD<T> &inflate_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, const Vector<uint> &dims) const
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
    TensorD<T> &squeeze(TensorD<T> &y) const
    {
        Vector<uint> new_dim;
        for (uint i = 0; i < dim().size(); ++i)
        {
            if (dim()[i] > 1)
            {
                new_dim.push_back(dim()[i]);
            }
        }

        y.deep_dim_copy(*this);
        y._dim->copy(new_dim);
        return y;
    }

    TensorD<T> &squeeze_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad) const
    {
        if (!x_grad.dim().equals_to(this->dim()))
        {
            x_grad.reset(this->dim(), TensorInit_Types::Zero);
        }

        x_grad.vector().copy(y_grad.vector());
        return x_grad;
    }

    TensorD<T> &subset(TensorD<T>& y, const Vector<uint>& dim, uint offset) const
    {
        uint size = dim.product();
        assert(offset + size <= this->size());
        
        y.reset(dim);
        y._vector->copy(this->vector(), offset, size); // deep copy
        return y;
    }

    SUGAR TensorD<T> subset(const Vector<uint>& dim, uint offset) const
    {
        TensorD<T> y;
        return subset(y, dim, offset);
    }

    TensorD<T> &subset_grad(const TensorD<T> &y, const TensorD<T> &y_grad, TensorD<T> &x_grad, const Vector<uint>& dim, uint offset) const
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

#pragma endregion
};

// template<> class TensorD<double>;
// template<> class TensorD<int>
