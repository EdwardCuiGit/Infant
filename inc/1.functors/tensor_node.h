#pragma once
#include "bin_func.h"
#include "un_func.h"

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

class TensorN : public TensorD<double>
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
    TensorD<double> grad;

    // not very good design to put momentum here since only parameter Tensors will use momentum
    // ideally should create Parameter sub class to inherit from Tensor
    TensorD<double> momentum;

    // all the FunctorNodes which will use this TensorNode's data as input
    FunctorNodePList down_functors;

    // what is output functor who generates data in forward() process
    // each Tensor only has one output functor
    // if output_functor == nullptr, this is overall external inputs
    Ptr<FunctorNode> output_functor = nullptr;
    bool is_print = false;
    bool is_param = false;
    bool is_auto_grad = true;
    std::string id; // not used, will be used to find given tensor, e.g., used for input tensor matching & assignment
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

    static Tensor deep_upgrade(const TensorD<double>& x);

    // be catious for these weak_upgrade/weak_downgrade, TensorD.vector & dim are both shared
    static Tensor weak_upgrade(const TensorD<double>& x);

    static TensorList weak_upgrade(const TensorDArray<double>& x);

    static TensorDArray<double> weak_data_downgrade(const TensorList& x);

    static void weak_both_downgrade(const TensorList& x, TensorDArray<double>& data, TensorDArray<double>& grad);

    inline Tensor reset(const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None)
    {
        // TODO: do we need to clear up_functors & down_functors
        this->get()->reset(dim, t);
        this->get()->grad.reset(dim, TensorInit_Types::Zero);
        return *this;
    }

    // refresh features, clear up grads
    inline void weak_copy(const Tensor& x)
    {
        this->data().weak_copy(x.data());
        this->grad().weak_copy(x.grad());
    }

    inline void deep_copy(const Tensor& x)
    {
        this->reset(x.dim());
        this->data().deep_copy(x.data());
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

    inline void print(std::ostream &os) const
    {
        // TODO
    }

    inline TensorD<double> &data()
    {
        return *this->get();
    }

    inline const TensorD<double> &data() const
    {
        return *this->get();
    }

    inline TensorD<double> &grad()
    {
        return this->get()->grad;
    }

    inline TensorD<double> &momentum()
    {
        return this->get()->momentum;
    }

    inline const TensorD<double> &grad() const
    {
        return this->get()->grad;
    }

    // hide access to internal data
    /*Vector<double>& vector()
    {
        return this->data().vector();
    }*/

protected:
    // 0. these are template functions that could call any Functor directly, no need below specific function names
    // create parameter variant function f<Combine>, f<Sum>, and then no need to create one new function after a new functor is built

    static void f(PFunctor func, const TensorList &x, TensorList &y);

    template <typename T, typename... Args>
    const Tensor f(const Tensor &x2, Args &&...args) const;

    template <typename T, typename... Args>
    Tensor& f_(const Tensor &x2, Args &&...args);

    template <typename T, typename... Args>
    const Tensor f(Args &&...args) const;

    template <typename T, typename... Args>
    Tensor& f_(Args &&...args);

    template <typename T, typename... Args>
    TensorList fm(Args &&...args) const;

    template <typename T, typename... Args>
    static void f(const TensorList &x, TensorList &y, Args &&...args);
public:
    // 1: below are binary ops
    USED inline Tensor add(const Tensor &x2, double alpha_x1 = 1.0, double alpha_x2 = 1.0, double beta = 0.0,
                      uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return f<Add>(x2, alpha_x1, alpha_x2, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor& add_(const Tensor &x2, double alpha_x1 = 1.0, double alpha_x2 = 1.0, double beta = 0.0,
                       uint first_match_dims = 0, int last_work_dims = -1)
    {
        return f_<Add>(x2, alpha_x1, alpha_x2, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor mul(const Tensor &x2, double alpha = 1.0, double beta = 0.0, uint first_match_dims = 0, int last_work_dims = -1) const
    {
        return f<Mul>(x2, alpha, beta, first_match_dims, last_work_dims);
    }

    USED inline Tensor& mul_(const Tensor &x2, double alpha = 1.0, double beta = 0.0, uint first_match_dims = 0, int last_work_dims = -1)
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

    // 2: below are unary ops
    USED inline Tensor linear(double alpha = 1.0, double beta = 0.0, int last_work_dims = -1) const
    {
        return f<Linear>(alpha, beta, last_work_dims);
    }

    inline Tensor& linear_(double alpha = 1.0, double beta = 0.0, int last_work_dims = -1)
    {
        return f_<Linear>(alpha, beta, last_work_dims);
    }

    USED inline Tensor pow(double n, double bias = 0, int last_work_dims = -1) const
    {
        return f<Pow>(n, bias, last_work_dims);
    }

    inline Tensor softmax(int last_work_dims = -1) const
    {
        return f<Softmax>(last_work_dims);
    }

    USED inline Tensor& softmax_(int last_work_dims = -1)
    {
        return f_<Softmax>(last_work_dims);
    }

    USED inline Tensor activation(Activation_Types act_type, int last_work_dims = -1) const
    {
        return f<Activation>(act_type, last_work_dims);
    }

    USED inline Tensor& activation_(Activation_Types act_type, int last_work_dims = -1)
    {
        return f_<Activation>(act_type, last_work_dims);
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

    USED inline Tensor im2col(uint groups, uint kernel_x, uint kernel_y, uint stride_x, uint stride_y, uint padding_x, uint padding_y) const
    {
        return f<Im2Col>(groups, kernel_x, kernel_y, stride_x, stride_y, padding_x, padding_y);
    }

    USED inline Tensor merge_dim(uint from, uint len) const
    {
        return f<MergeDim>(from, len);
    }

    inline Tensor inflate(const Vector<uint>& dims) const
    {
        return f<Inflate>(dims);
    }

    USED inline Tensor squeeze() const
    {
        return f<Squeeze>();
    }

    USED inline Tensor subset(const Vector<uint>& dim, uint offset) const
    {
        return f<Subset>(dim, offset);
    }

    // divide one tensor to be multiple tensors
    USED inline TensorList divide(uint first_match_dims = 1) const
    {
        return fm<Divide>(first_match_dims);
    }

    USED static inline Tensor combine(const TensorList &x, const Vector<uint>& first_dims = {})
    {
        TensorList y;
        f<Combine>(x, y, first_dims);
        return y[0];
    }

    // below are high level operators
    inline Tensor fc(uint input_dim, uint output_dim, bool has_bias = false, 
        TensorInit_Types w_type = TensorInit_Types::Gaussian, TensorInit_Types b_type = TensorInit_Types::Zero) const;

    inline Tensor conv2d(uint in_channels, uint out_channels, uint kernel_x = 3, uint kernel_y = 3, 
        uint stride_x = 1, uint stride_y = 1, uint padding_x = 0, uint padding_y = 0, 
        uint groups = 1, bool has_bias = true, TensorInit_Types k_type = TensorInit_Types::Gaussian,
        TensorInit_Types b_type = TensorInit_Types::Zero) const;

    inline Tensor pool2d(Pooling_Types pt, uint kernel_x, uint kernel_y, uint stride_x = 1, 
        uint stride_y = 1, uint padding_x = 0, uint padding_y = 0) const;
};
