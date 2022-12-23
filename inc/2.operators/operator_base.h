#pragma once

#include "../1.functors/tensor_node.h"
#include <map>

// TODO: add required shape for both input & outputs
/*class Operator
{
protected:
    bool _is_train = false; // TODO: not set yet
    std::string _id;
    std::map<std::string, Ptr<Parameter>> _p; // all parameters
    uint _macc = 0;                                       // not used

public:
    inline ParameterPtr create_param(const std::string& id, const Vector<uint>& dims, TensorInit_Types init_type)
    {
        return _p[id] = std::make_shared<Parameter>(id, dims, init_type);
    }

    virtual bool has_loss() const
    {
        return false;
    }

    void set_train(bool yes = true)
    {
        _is_train = yes;
    }

    // y will be init size inside forward function
    virtual void forward(const TensorD<double> &x, TensorD<double> &y) const = 0;

    virtual void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad) = 0;

    virtual uint input_tensor_count() const
    {
        return 1;
    }

    virtual uint output_tensor_count() const
    {
        return 1;
    }

    uint parameter_count() const
    {
        uint total = 0;
        for (auto p : _p)
        {
            total += p.second->size();
        }

        return total;
    }

    uint macc() const
    {
        return _macc;
    }

    uint peak_memory() const
    {
    }

    const std::map<std::string, Ptr<Parameter>> *get_params() const
    {
        return &_p;
    }
};

class OperatorMio : public Operator
{
private:
    OVERRIDE void forward(const TensorD<double> &x, TensorD<double> &y) const
    {
        assert(false);
    }

    OVERRIDE void backward(const TensorD<double> &x, const TensorD<double> &y, const TensorD<double> &y_grad, TensorD<double> &x_grad)
    {
        assert(false);
    }

public:
    //virtual void forward_mio(const VectorBase<const Ptr<const Tensor<double>>>& x,
    //    const VectorBase<const Ptr<Tensor<double>>>& y) const
    virtual void forward_mio(const TensorList &x, TensorList &y) const
    {
        assert(false);
    }

    virtual void backward_mio(const TensorList &x, const TensorList &y, const TensorList &y_grad,
                              TensorList &x_grad)
    {
        assert(false);
    }
};
enum OpConfig_Keys
{
    HAS_BIAS,
    INIT_W,
    INIT_B,
};

class OperatorConfig
{
public:
    bool get_bool(OpConfig_Keys key) const
    {
        // TODO
        assert(false);
    }

    int get_int(OpConfig_Keys key) const
    {
        // TODO
        assert(false);
    }
};*/

/*class Parameter : public Tensor
{
public:
    TensorInit_Types init_type; 
    bool save_to_model;         // TODO: not used
    TensorD<double> momentum;          // TODO: not used

public:
    Parameter(){}
    Parameter(const std::string &id, const Vector<uint> &dim, TensorInit_Types t = TensorInit_Types::None)
        : Tensor(dim, t, id, true), save_to_model(true), init_type(t)
    {
    }
};

typedef Ptr<Parameter> PParameter;
typedef Array<PParameter> PParameterList;

// TODO: not used
class ParameterGroup
{
public:
    PParameterList params;
    std::map<std::string, double> configs; // update strategy is here
};
*/
// Learnable Operator: added parameters
// TODO: add required shape for both input & outputs
class Operator : public Functor
{
protected:
    //mutable PParameterList _p; // all parameters
    mutable TensorList _p;
    Operator(const std::string& type) : Functor(type){}

public:
    inline Tensor create_param(const std::string &id, const Vector<uint> &dims, TensorInit_Types init_type)
    {
        Tensor p(dims, init_type, id, true);
        _p.push_back(p);
        return p;
    }

    // used for serialization
    virtual void load(const std::istream &input)
    {}

    virtual void save(std::ostream &output) const
    {}

    // auto_grad, not used below, pls do not override this
    SEALED virtual void backward(TensorList &x, const TensorList &y) override
    {
        assert(false);
    }

    uint parameter_count() const
    {
        uint total = 0;
        for (auto p : _p)
        {
            total += p.data().size();
        }

        return total;
    }

    TensorList parameters()
    {
        return _p;
    }

    /*
    virtual bool has_loss() const
    {
        return false;
    }*/
};

// Unary Learnable Operator
class UnOp : public Operator
{
public:
    UnOp() : Operator("UnOp"){}
    virtual void forward(const Tensor &x, Tensor &y) const = 0;//note: could be protected

    virtual void forward(const TensorList &x, TensorList &y) const override
    {
        assert(x.size() == 1);
        assert(y.size() == 1);
        // TODO: shall we keep y nullptr first?

        forward(x[0], y[0]);
    }
};
