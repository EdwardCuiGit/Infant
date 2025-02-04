#pragma once
#include "functor.h"

// TODO: NetworkNode's memory cost is huge now
// which owns data: FunctorNode.func, TensorNode.data, TensorNode.grad, FunctorGraph._functors, FunctorGraph._tensors
// which NOT owns data: inputs, outputs, up_functors, down_functors;
class FunctorNode;
typedef Ptr<FunctorNode> PFunctorNode;
typedef Array<PFunctorNode> FunctorNodePList;
struct FunctorNode
{
public:
    // this is core of FunctorNode: Functor, which is calculation module
    PFunctor func; // owns func data

    // below are inputs/outputs tensors for this Functor
    // input TensorD pointer's list, these pointers will point to other node's output, not owning data
    // starting inputs will be owned by FunctorGraph
    TensorList inputs;
    // output TensorD pointer's list, this FunctorNode will own output TensorD
    TensorList outputs;

    // after up functors are all executed, we will execute current functor
    // why not depend on input tensors to be filled: that's because multiple functors may share one tensor due to in-place functors, e.g., add_
    FunctorNodePList up_functors;
    FunctorNodePList down_functors;

    //std::string id;
    uint index;
};

// Note: actually this could support multiple connected graphs, i.e., this is a list of graphs.
class FunctorGraph : Functor
{
friend class Tensor;
friend class TestFunctorGraph;
private:
    FunctorNodePList _functors;
    TensorList _tensors;
    TensorList _params;
    mutable TensorList _input_tensors;
    TensorList _output_tensors;

    FunctorGraph() : Functor("FunctorGraph"){}

    // topological order traverse
    // each node will be executed only all the up_stream nodes are executed, i.e., all its inputs are ready
    // note: it's DAG graph
    void forward_traverse(const std::function<void(FunctorNode &)> &func) const;

    void backward_traverse(const std::function<void(FunctorNode &)> &func) const;
    virtual void forward(const TensorList &x, TensorList &y) const override;
    virtual void backward(TensorList &_, const TensorList &y) override;

    void add(PFunctor func, const TensorList &inputs, const TensorList &outputs);

    NOTIMPLEMENTED void prune(const Tensor& start_tensor);

public:
    // note: all the external inputs, needs to assign value by orders
    // note: needs to ensure all the tensors in x are in the graph
    void set_inputs(const TensorList& x)
    {
        this->_input_tensors = x;
    }

    //note: all the output tensors
    void set_outputs(const TensorList& y)
    {
        this->_output_tensors = y;
    }

    void reset()
    {
        this->_tensors.clear();
        this->_functors.clear();
        this->_input_tensors.clear();
        this->_output_tensors.clear();
        this->_params.clear();
    }

    TensorList forward(const TensorList& x) const;
    void backward(const TensorList& output_tensor_grads);
    NOTIMPLEMENTED void print(std::ostream &os) const;

    // TODO: what to save: better to save as ONNX format
        // functor type/variables & parameters, dependent tensors, etc. this is similar to FunctorGraph but w/o feature map
        // which tensors are input & output tensors also need to record
        // in serialization, each tensor needs to be represented & searched by internal auto-incr-id
        // we may also need to prune useless functors and tensors
    // TODO: how to support to save 2 model files?
    NOTIMPLEMENTED void save_model(const std::string& file_name) const;
    NOTIMPLEMENTED bool load_model(const std::string& file_name);

    // after one iteration, zero out all feature maps
    void zero_features()
    {
        for (Tensor t : _tensors)
        {
            // note: this will not zero out parameter values
            if (!t.is_param())
            {
                t.data().clear();
            }
        }
    }

    // zero grad for all parameters & features
    void zero_grads()
    {
        // this is covering above parameters
        for (Tensor t : _tensors)
        {
            t.grad().clear();
        }
    }

private:
    static FunctorGraph _g;
    bool _start_auto_grad = false;

public:
    static FunctorGraph& singleton()
    {
        return _g;
    }

    void stop_auto_grad()
    {
        _start_auto_grad = false;
    }

    void start_auto_grad()
    {
        _start_auto_grad = true;
    }

    bool is_auto_grad()
    {
        return _start_auto_grad;
    }

    TensorList& params()
    {
        return _params;
    }
};

FunctorGraph FunctorGraph::_g;