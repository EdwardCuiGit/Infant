#pragma once
#include "bin_func.h"
#include "un_func.h"
#include "functor_graph.h"

/*
there are two options to implement pointer based Tensor:
1) use boost::enable_shared_from_this<TensorN>
pros: one class only
cons: can't assign new pointer to self
2) use 2 classes, second inherit Ptr
pros: could assign self
cons: 2 classes
*/
class FunctorNode;
typedef Ptr<FunctorNode> PFunctorNode;
typedef Array<PFunctorNode> FunctorNodePList;

class TensorN : public TensorD<float>
{
    friend class Tensor;
    friend class FunctorGraph;

protected:
    // grad of major tensor data, totally the same shape/dim
    // note: some TensorNode no need to calculate and track grad

    // if it's used by multiple downstream node, we need to sum up them
    /* if one node's output is used as input for multiple downstream nodes, should we sum up these multiple nodes' grads?? yes
    e.g., y1, y2 = f(x1, x2, x3), z1, z2 = g(x1, y1, x4), t1 = h(y1, z1), x2 = m(x1)
    and then we needs to store input_grad data for input node separately, and sum up later
    this is 1:1 map to inputs, if the referenced output tensor just has one downstream node, actually input_grad == output_grad
    but else, we need to sum up to output_grad;
    TODO: no need to store data if it's only user for up node's output, even not, we could create thread-safe add_ func
    Actually in this new structure, we just need to add_to grad during multiple downstream functor's backward() func*/
    TensorD<float> grad;

    // not very good design to put momentum here since only parameter Tensors will use momentum
    // ideally should create Parameter sub class to inherit from Tensor
    TensorD<float> momentum;

    // all the FunctorNodes which will use this TensorNode's data as input
    FunctorNodePList down_functors;

    // what is output functor who generates data in forward() process
    // each Tensor only has one output functor
    // if output_functor == nullptr, this is overall external inputs
    Ptr<FunctorNode> output_functor = nullptr;
    bool is_print = false;
    bool is_param = false;
    bool is_auto_grad = true;
    std::string id; // will be used to find given tensor, e.g., used for input tensor matching & assignment
};

class Tensor : public Ptr<TensorN>
{
public:
    Tensor()
    {
        this->Ptr<TensorN>::reset(new TensorN());
    }

    Tensor(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None, const std::string &id = "", bool is_param = false)
    {
        this->Ptr<TensorN>::reset(new TensorN());
        this->reset(dim, t);
        this->get()->id = id;
        this->get()->is_param = is_param;
    }

    Tensor(const Vector<uint> &dim, const Vector<float>& data, const std::string &id = "", bool is_param = false)
    {
        this->Ptr<TensorN>::reset(new TensorN());
        this->reset(dim, data);
        this->get()->id = id;
        this->get()->is_param = is_param;
    }

    static Tensor Deep_Upgrade(const TensorD<float> &x);

    // be catious for these weak_upgrade/weak_downgrade, TensorD.vector & dim are both shared
    static Tensor Weak_Upgrade(const TensorD<float> &x);

    static TensorList Weak_Upgrade(const TensorDArray<float> &x);

    static TensorDArray<float> Weak_Data_Downgrade(const TensorList &x);

    static void Weak_Both_Downgrade(const TensorList &x, TensorDArray<float> &data, TensorDArray<float> &grad);

    inline Tensor reset(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None)
    {
        // TODO: do we need to clear up_functors & down_functors
        this->get()->reset(dim, t);
        this->get()->grad.reset(dim, TensorInit_Types::Zero);
        return *this;
    }

    inline void reset(const Vector<uint> &dim, const Vector<float>& data)
    {
        this->get()->reset(dim, data);;
        this->get()->grad.reset(dim, TensorInit_Types::Zero);
    }

    // refresh features, clear up grads
    inline void weak_copy(const Tensor &x)
    {
        this->data().weak_copy(x.data());
        this->grad().weak_copy(x.grad());
    }

    inline static Tensor From_Data(const TensorD<float> &x)
    {
        Tensor res;
        res.reset(x.dim());
        res.data().weak_copy(x);
        return res;
    }

    USED inline Tensor encode_by_dict(const Tensor &x2) const
    {
        return f<EncodeByDict>(x2);
    }

    USED inline Tensor search_by_dict(const Tensor &x2, int padding_id = -1) const
    {
        return f<SearchByDict>(x2, padding_id);
    }

    USED inline Tensor decode__() const
    {
        TensorD<float> res;
        this->data().decode__(res);
        return Tensor::From_Data(res);
    }

    // only stores TensorD and id/is_**, momentum, down_functors/output_functor are not stored 
    inline void save(std::ostream& o) const
    {
        StringUtil::write_string(o, "id", this->id());
        StringUtil::write_uint(o, "is_auto_grad", (uint)this->is_auto_grad());
        StringUtil::write_uint(o, "is_param", (uint)this->is_param());
        StringUtil::write_uint(o, "is_print", (uint)this->is_print());
        this->data().save(o);
    }

    // this is pair function of above save()
    inline void load(std::istream& i)
    {
        this->get()->id = StringUtil::read_string(i, "id");
        this->get()->is_auto_grad = (bool)StringUtil::read_uint(i, "is_auto_grad");
        this->get()->is_param= (bool)StringUtil::read_uint(i, "is_param");
        this->get()->is_print= (bool)StringUtil::read_uint(i, "is_print");

        this->data().load(i);
        this->get()->grad.reset(this->dim(), TensorInit_Types::Zero);
    }


    inline uint shape() const
    {
        return this->get()->shape();
    }

    inline uint size() const
    {
        return this->get()->size();
    }

    inline const Vector<uint> &dim() const
    {
        return this->get()->dim();
    }

    inline uint size_to_dim(uint size, bool forward = true) const
    {
        return this->get()->size_to_dim(size, forward);
    }

    // note: only check data, don't check grad, momentum, id, etc.
    inline bool equals_to(const Tensor &x, float noise_level = 0.00001) const
    {
        return this->get()->equals_to(x.data(), noise_level);
    }

    inline bool equals_to(const Vector<uint>& dim, const Vector<float>& vector, float noise_level = 0.00001) const
    {
        return this->get()->equals_to(dim, vector, noise_level);
    }

    inline const std::string &id() const
    {
        return this->get()->id;
    }

    inline void set_print()
    {
        this->get()->is_print = true;
    }

    inline bool is_print() const
    {
        return this->get()->is_print;
    }

    inline bool is_param() const
    {
        return this->get()->is_param;
    }

    inline bool is_auto_grad() const
    {
        return this->get()->is_auto_grad;
    }

    inline void set_auto_grad(bool yes = true)
    {
        this->get()->is_auto_grad = yes;
    }

    inline TensorD<float> &data()
    {
        return *this->get();
    }

    inline const TensorD<float> &data() const
    {
        return *this->get();
    }

    inline TensorD<float> &grad()
    {
        return this->get()->grad;
    }

    inline TensorD<float> &momentum()
    {
        return this->get()->momentum;
    }

    inline const TensorD<float> &grad() const
    {
        return this->get()->grad;
    }

public:
    // 0. these are template functions that could call any Functor directly, no need below specific function names
    // create parameter variant function f<Combine>, f<Sum>, and then no need to create one new function after a new functor is built

    /*static void fmm(PFunctor func, const TensorList &x, TensorList &y);

    template <typename T, typename... Args>
    const Tensor f(const Tensor &x2, Args &&...args) const;

    template <typename T, typename... Args>
    Tensor &f_(const Tensor &x2, Args &&...args);

    template <typename T, typename... Args>
    const Tensor f(Args &&...args) const;

    template <typename T, typename... Args>
    Tensor &f_(Args &&...args);

    template <typename T, typename... Args>
    TensorList fm(Args &&...args) const;

    template <typename T, typename... Args>
    static void fmmn(const TensorList &x, TensorList &y, Args &&...args);*/

    // 0. these are template functions that could call any Functor directly, no need below specific function names
    template <typename T, typename... Args>
    const Tensor f(const Tensor &x2, Args &&...args) const
    {
        PFunctor func = std::make_shared<T>(args...);
        TensorList x{*this, x2}, y{Tensor()};
        // LOG_INFO("f: " << func->type());
        func->forward(x, y);
        y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

        if (FunctorGraph::singleton().is_auto_grad() && this->is_auto_grad())
        {
            FunctorGraph::singleton().add(func, x, y);
        }

        return y[0];
    }

    template <typename T, typename... Args>
    Tensor &f_(const Tensor &x2, Args &&...args)
    {
        PFunctor func = std::make_shared<T>(args...);
        TensorList x{*this, x2}, y{Tensor()};
        // LOG_INFO("f_: " << func->type());
        func->forward(x, y);
        y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

        if (FunctorGraph::singleton().is_auto_grad() && this->is_auto_grad())
        {
            FunctorGraph::singleton().add(func, x, y);
        }

        // put it after _g.add to ensure original this->get() is not released
        this->Ptr<TensorN>::operator=(y[0]);
        return *this;
    }

    static void fmm(PFunctor func, const TensorList &x, TensorList &y)
    {
        // LOG_INFO("fmm: " << func->type());
        func->forward(x, y);
        assert(x.size() > 0);
        if (FunctorGraph::singleton().is_auto_grad() && x[0].is_auto_grad())
        {
            FunctorGraph::singleton().add(func, x, y);
        }
    }

    template <typename T, typename... Args>
    const Tensor f(Args &&...args) const
    {
        PFunctor func = std::make_shared<T>(args...);
        TensorList x{*this}, y{Tensor()};

        fmm(func, x, y);
        y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

        return y[0];
    }

    template <typename T, typename... Args>
    Tensor &f_(Args &&...args)
    {
        PFunctor func = std::make_shared<T>(args...);
        TensorList x{*this}, y{Tensor()};
        fmm(func, x, y);
        y[0].grad().reset(y[0].data().dim(), TensorInit_Types::Zero);

        // put it after _g.add to ensure original this->get() is not released
        this->Ptr<TensorN>::operator=(y[0]);
        return *this;
    }

    template <typename T, typename... Args>
    TensorList fm(Args &&...args) const
    {
        PFunctor func = std::make_shared<T>(args...);
        TensorList x{*this}, y;

        fmm(func, x, y);

        return y;
    }

    template <typename T, typename... Args>
    static void fmmn(const TensorList &x, TensorList &y, Args &&...args)
    {
        PFunctor func = std::make_shared<T>(args...);
        // LOG_INFO("fmmn: " << func->type());
        func->forward(x, y);
    }


public:
    // 1: below are binary ops
    USED inline Tensor add(const Tensor &x2, float alpha_x1 = 1.0, float alpha_x2 = 1.0, float beta = 0.0,
                           uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return f<Add>(x2, alpha_x1, alpha_x2, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor &add_(const Tensor &x2, float alpha_x1 = 1.0, float alpha_x2 = 1.0, float beta = 0.0,
                             uint first_match_dims = 0, int last_work_dims = -1)
    {
        return f_<Add>(x2, alpha_x1, alpha_x2, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor mul(const Tensor &x2, float alpha = 1.0, float beta = 0.0, uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return f<Mul>(x2, alpha, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor &mul_(const Tensor &x2, float alpha = 1.0, float beta = 0.0, uint first_match_dims = 0, int last_work_dims = -1)
    {
        return f_<Mul>(x2, alpha, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor dot(const Tensor &x2, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        return f<Dot>(x2, first_match_dims, last_work_dims);
    }

    inline Tensor mse(const Tensor &x2, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        return f<Mse>(x2, first_match_dims, last_work_dims);
    }

    inline Tensor ce(const Tensor &x2, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        return f<Ce>(x2, first_match_dims, last_work_dims);
    }

    inline Tensor euclidean(const Tensor &x2, uint first_match_dims = 0, int last_work_dims = 1) const
    {
        return f<Euclidean>(x2, first_match_dims, last_work_dims);
    }

    // note: map function can't be implemented by cuda, can be used for protoyping in cpu
    // 2: below are unary ops
    USED inline Tensor map(const std::function<float(float)>& func, 
    const std::function<float(float)>& func_grad) const
    {
        return f<MapFunctor>(func, func_grad);
    }

    USED inline Tensor map_(const std::function<float(float)>& func, 
    const std::function<float(float)>& func_grad)
    {
        return f_<MapFunctor>(func, func_grad);
    }

    USED inline Tensor map(const std::function<void(const Vector<float>&, uint, uint, Vector<float>&  )>& func,
    const std::function<void(const Vector<float>&, uint, uint, const Vector<float>&, Vector<float>&)>& func_grad,
    int last_work_dims = -1) const
    {
        return f<MapFunctor>(func, func_grad, last_work_dims);
    }

    USED inline Tensor &map_(const std::function<void(const Vector<float>&, uint, uint, Vector<float>&)>& func, 
    const std::function<void(const Vector<float>&, uint, uint, const Vector<float>&, Vector<float>&)>& func_grad,
    int last_work_dims = -1)
    {
        return f_<MapFunctor>(func, func_grad, last_work_dims);
    }

    USED inline Tensor linear(float alpha = 1.0, float beta = 0.0, int last_work_dims = -1) const
    {
        return f<Linear>(alpha, beta, last_work_dims);
    }

    inline Tensor &linear_(float alpha = 1.0, float beta = 0.0, int last_work_dims = -1)
    {
        return f_<Linear>(alpha, beta, last_work_dims);
    }

    USED inline Tensor pow(float n, float bias = 0, int last_work_dims = -1) const
    {
        return f<Pow>(n, bias, last_work_dims);
    }

    USED inline Tensor ln(float bias = 0, int last_work_dims = -1) const
    {
        return f<Ln>(bias, last_work_dims);
    }

    inline Tensor &ln_(float bias = 0, int last_work_dims = -1)
    {
        return f_<Ln>(bias, last_work_dims);
    }

    inline Tensor softmax(int last_work_dims = -1) const
    {
        return f<Softmax>(last_work_dims);
    }

    USED inline Tensor &softmax_(int last_work_dims = -1)
    {
        return f_<Softmax>(last_work_dims);
    }

    USED inline Tensor activation(Activation_Types act_type, int last_work_dims = -1) const
    {
        return f<Activation>(act_type, last_work_dims);
    }

    USED inline Tensor &activation_(Activation_Types act_type, int last_work_dims = -1)
    {
        return f_<Activation>(act_type, last_work_dims);
    }

    USED inline Tensor rms_norm(float gamma= 1.0f, int last_work_dims = -1) const
    {
        return f<RmsNorm>(gamma, last_work_dims);
    }

    USED inline Tensor replace(float cond_value, float if_value, float else_value) const
    {
        return f<Replace>(cond_value, if_value, else_value);
    }

    USED inline Tensor insert(uint pos, float value, int last_work_dims = -1) const
    {
        return f<Insert>(pos, value, last_work_dims);
    }

    USED inline Tensor merge(const Tensor &x2, int dim = 0) const
    {
        return f<Merge>(x2, dim);
    }

    inline Tensor sum(int last_work_dims = -1) const
    {
        return f<Sum>(last_work_dims);
    }

    inline Tensor avg(int last_work_dims = -1) const
    {
        return f<Avg>(last_work_dims);
    }

    USED inline Tensor var(bool biased = false, int last_work_dims = -1) const
    {
        return f<Var>(biased, last_work_dims);
    }

    inline Tensor max(int last_work_dims = -1) const
    {
        return f<Max>(last_work_dims);
    }

    inline Tensor min(int last_work_dims = -1) const
    {
        return f<Min>(last_work_dims);
    }

    // 2.1 below are tensor manipulation ops
    USED inline Tensor swap(uint first_dim, uint second_dim) const
    {
        return f<Swap>(first_dim, second_dim);
    }

    USED inline Tensor move_forward(uint move_from, uint move_len, uint move_to) const
    {
        return f<MoveForward>(move_from, move_len, move_to);
    }

    USED inline Tensor move_forward_(uint move_from, uint move_len, uint move_to)
    {
        return f_<MoveForward>(move_from, move_len, move_to);
    }

    USED inline Tensor im2col(uint groups, uint kernel_x, uint kernel_y, uint stride_x, uint stride_y, uint padding_x, uint padding_y) const
    {
        return f<Im2Col>(groups, kernel_x, kernel_y, stride_x, stride_y, padding_x, padding_y);
    }

    USED inline Tensor merge_dim(uint from, uint len) const
    {
        return f<MergeDim>(from, len);
    }

    inline Tensor inflate(const Vector<uint> &dims) const
    {
        return f<Inflate>(dims);
    }

    USED inline Tensor squeeze(int dim = -1) const
    {
        return f<Squeeze>(dim);
    }

    USED inline Tensor unsqueeze(uint dim) const
    {
        return f<Unsqueeze>(dim);
    }

    USED inline Tensor reshape(const Vector<uint> &dim) const
    {
        return f<Reshape>(dim);
    }

    USED inline Tensor subset(const Vector<uint> &dim, uint offset) const
    {
        return f<Subset>(dim, offset);
    }

    USED inline Tensor dropout(float p) const
    {
        return f<Dropout>(p);
    }

    // divide one tensor to be multiple tensors
    USED inline TensorList divide(uint first_match_dims = 1) const
    {
        return fm<Divide>(first_match_dims);
    }

    USED static inline Tensor combine(const TensorList &x, const Vector<uint> &first_dims = {})
    {
        TensorList y;
        fmmn<Combine>(x, y, first_dims);
        return y[0];
    }

    // note: dim_to_inc means which dim to +1
    inline Tensor append_(const Tensor &x, uint dim_to_inc = 0)
    {
        return f_<Append>(x, dim_to_inc);
    }

    USED inline TensorList where(CompareTypes type, float v) const
    {
        return fm<Where>(type, v);
    }

    USED inline TensorList topk(uint k) const
    {
        return fm<TopK>(k);
    }

    USED inline Tensor index(const TensorList &indices, bool cross = false) const
    {
        auto func = std::make_shared<Index>(cross);
        TensorList x, y;
        x.push_back(*this);
        x.append(indices);
        fmm(func, x, y);
        return y[0];
    }

    USED inline Tensor assign(const Tensor &values, const Tensor &indices) const
    {
        auto func = std::make_shared<Assign>();
        TensorList x, y;
        x.push_back(*this);
        x.push_back(values);
        x.push_back(indices);
        fmm(func, x, y);
        return y[0];
    }

    /*    // below are high level operators
        inline Tensor fc(uint input_dim, uint output_dim, bool has_bias = false,
                         TensorInit_Types w_type = TensorInit_Types::Gaussian, TensorInit_Types b_type = TensorInit_Types::Zero) const;

        inline Tensor conv2d(uint in_channels, uint out_channels, uint kernel_x = 3, uint kernel_y = 3,
                             uint stride_x = 1, uint stride_y = 1, uint padding_x = 0, uint padding_y = 0,
                             uint groups = 1, bool has_bias = true, TensorInit_Types k_type = TensorInit_Types::Gaussian,
                             TensorInit_Types b_type = TensorInit_Types::Zero) const;

        inline Tensor pool2d(Pooling_Types pt, uint kernel_x, uint kernel_y, uint stride_x = 1,
                             uint stride_y = 1, uint padding_x = 0, uint padding_y = 0) const;

        inline Tensor layer_norm(const Coefficients::LayerNorm& c) const;
        inline Tensor multi_head_attention(const Coefficients::SelfAttention& c) const;*/
};
