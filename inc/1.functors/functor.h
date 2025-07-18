#pragma once
#include "inc/0.tensors/tensor.h"

/* Functor/UpOp/Tensor are used to build dynamic execution graph
Functor is the unit level compute node, no learnable parameters
Tensor is unit level data node, used to store features or learnable parameters
UpOp is compute node with auto-grad, mostly has learnable parameters
*/
class Functor;
typedef Ptr<Functor> PFunctor;
class Tensor;
typedef Array<Tensor> TensorList;

class Functor
{
protected:
    std::string _id; // not used, which is used for visualization
    std::string _type;
    // TODO: this is a waste of memory since the whole env is almost the same for is_train
    // bool _is_train = false; // some operators behave different in train & inference
public:
    Functor(const std::string& type) : _type(type)
    {}
    // make it virtual so that inheritance's dctor works well
    ~Functor(){}
    // Note: for the output y Tensorlist, we will not reserve # of tensors before calling forward
    virtual void forward(const TensorList &x, TensorList &y) const = 0;

    // the major input y_grad is in y[i].grad, the major output x_grad is in x[i].grad
    // TODO: Note: all the grad value needs to add to x[i].grad(), not assign to, since each x may need have multiple sources of grad
    virtual void backward(TensorList &x, const TensorList &y) = 0;

    int check_isnan(const TensorList &x) const;

    virtual uint input_tensor_count() const
    {
        return 0;
    }

    virtual uint output_tensor_count() const
    {
        return 0;
    }

    virtual uint macc() const
    {
        return 0; // TODO
    }

    virtual uint peak_memory() const
    {
        return 0; // TODO
    }

    std::string type() const
    {
        return _type;
    }

    /*void set_train(bool yes = true)
    {
        _is_train = yes;
    }*/

    // whether this functor/operator's forward func will manipulate class memeber variables
    // if is_const is true, we no need to rebuild graph in every iteration; i.e., all the operator parameters are const
    // one example is Norm, which needs to store mean/var during runtime, so we need to run operator::forward every iteration
    // note: so far, every operator which has child non const operator, we need to explictly mark it as non const
    // note: we'd better have compile time check
    virtual bool is_const() const
    {
        return true;
    }
};

/*class GFunctor : public Functor
{
private:
    Vector<float> _params;
    std::function<void(const TensorList &x, TensorList &y)> _forward;
    std::function<void(TensorList &x, const TensorList &y)> _backward;
public:
    GFunctor(const std::string& type, const Vector<float>& params, 
    std::function<void(const TensorList &x, TensorList &y)> forward, 
    std::function<void(TensorList &x, const TensorList &y)> backward)
     : Functor(type), _params(params), _forward(forward), _backward(backward)
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override
    {
        _forward(x, y);
    }


    virtual void backward(TensorList &x, const TensorList &y) override
    {
        _backward(x, y);
    }

};
*/

// Combine N same sub-tensors to be one
// note: perf tuning: no need to physically allocate memories
class Combine : public Functor
{
private:
    Vector<uint> _first_dims;
public:
    Combine(const Vector<uint>& first_dims ={}) : Functor("Combine"), _first_dims(first_dims)
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint output_tensor_count() const override
    {
        return 1;
    }
};
// Implements TensorD<T>::where operation
// Returns a tensor with elements from x where condition is true, and elements from y where condition is false
CURSOR class Where : public Functor 
{
private:
    CompareTypes _type;
    float _v;
public:
    Where(CompareTypes type, float v) : Functor("Where"), _type(type), _v(v)
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint input_tensor_count() const override
    {
        return 1;
    }
};

CURSOR class TopK : public Functor
{
private:
    uint _k;
public:
    TopK(uint k) : Functor("TopK"), _k(k)
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint output_tensor_count() const override
    {
        return 2;
    }

    virtual uint input_tensor_count() const override
    {
        return 1;
    }
};

CURSOR class Index : public Functor
{
private:
    bool _cross;
public:
    Index(bool cross = false) : Functor("Index"), _cross(cross)
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;

    virtual uint output_tensor_count() const override
    {
        return 1;
    }
};

CURSOR class Assign : public Functor
{
public:
    Assign() : Functor("Assign")
    {
    }

    virtual void forward(const TensorList &x, TensorList &y) const override;

    virtual void backward(TensorList &x, const TensorList &y) override;
    
    virtual uint input_tensor_count() const override
    {
        return 3;
    }

    virtual uint output_tensor_count() const override
    {
        return 1;
    }
};