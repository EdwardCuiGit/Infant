#pragma once
#include "../../inc/0.tensors/tensor.h"

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
    bool _is_train = false; // some operators behave different in train & inference
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

    void set_train(bool yes = true)
    {
        _is_train = yes;
    }
};

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
