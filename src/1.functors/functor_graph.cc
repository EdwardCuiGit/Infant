#pragma once
#include "../../inc/1.functors/functor_graph.h"
#include "../../inc/1.functors/tensor_node.h"
#include <queue>

// the graph is built by add(), and link input tensors into the graph, and no need to add_tensor directly to the graph
void FunctorGraph::add(PFunctor func, const TensorList &inputs, const TensorList &outputs)
{
    // create Functor
    auto node = std::make_shared<FunctorNode>();
    node->func = func;
    node->inputs = inputs;
    node->outputs = outputs;
    node->index = _functors.size();
    _functors.push_back(node);

    // add new input/output tensors to _tensors
    // note: whether two tensor are the same is based on smart_pointer
    for (Tensor t : inputs)
    {
        if (!_tensors.contains(t))
        {
            _tensors.push_back(t);
            if (t.is_param())
            {
                _params.push_back(t);
            }

            // note: this tensor should be input features from external
            assert(t->output_functor == nullptr);
        }
    }

    for (Tensor t : outputs)
    {
        if (!_tensors.contains(t))
        {
            _tensors.push_back(t);
        }
        else
        {
            // note: we need to ensure tensors in outputs are newly created tensors
            // and this will ensure it's not cyclic graph
            assert(false);
        }
    }

    // append each input tensor's used_functors = func;
    // Tensor->down_functors not used
    for (Tensor t : inputs)
    {
        //t->down_functors.push_back(node);
        if (t->output_functor != nullptr)
        {
            node->up_functors.push_back(t->output_functor);
            t->output_functor->down_functors.push_back(node);
        }
    }

    for (Tensor t : outputs)
    {
        assert(t->output_functor == nullptr);
        t->output_functor = node;
    }
}

/*// for new iteration, we need to set new input features and labels
// these are set by id matching
void FunctorGraph::set_inputs(const TensorList &x) const
{
    for (auto t : x)
    {
        int index = this->_tensors.find([t](Tensor t1)->bool{return t.id() == t1.id();});
        assert(index < _tensors.size());
        this->_tensors[index].weak_copy(t);
    }
}

// Note: this is not working as newly created output tensors won't have y ids
void FunctorGraph::set_outputs(TensorList &y) const
{
    for (uint i = 0; i < y.size(); ++i)
    {
        int index = this->_tensors.find([y, i](Tensor t1)->bool{return y[i].id() == t1.id();});
        assert(index < _tensors.size());
        y[i].weak_copy(_tensors[index]);
    }
}*/

void FunctorGraph::print(std::ostream &os) const
{
    for (auto t : _tensors)
    {
        if (t.is_print())
        {
            t.print(os);
            //os << "\n";
        }
    }
}

void FunctorGraph::forward(const TensorList &x, TensorList &y) const
{
    y = this->forward(x);
}

TensorList FunctorGraph::forward(const TensorList &x) const
{
    // first, set all inputs, look for matched input tensors by order directly
    assert(x.size() == _input_tensors.size());
    for (uint i = 0; i < x.size(); ++i)
    {
        _input_tensors[i].weak_copy(x[i]);
    }

    // TODO: traverse the whole graph, only execute these whose inputs are ready, no need to consider parameters tensor
    forward_traverse([](FunctorNode &node) -> void {
        PFunctor func = node.func;
        func->forward(node.inputs, node.outputs);
        // TODO: no need to run loss ops in inference stage
        // TODO: this support multiple losses
        /*if (LossBase *lop = dynamic_cast<LossBase *>(mop))
        {
            double loss = lop->get_loss();
            // TODO: move this code to trainer
            LOG_INFO("LOSS:" << loss);
        }*/
    });

    return _output_tensors;
}

void FunctorGraph::forward_traverse(const std::function<void(FunctorNode &)> &func) const
{
    // look for starting functors, which has no input functors;
    std::queue<FunctorNode *> queue;
    Vector<int> status(_functors.size());
    for (uint i = 0; i < _functors.size(); ++i)
    {
        status[i] = _functors[i]->up_functors.size();
        if (status[i] == 0)
        {
            queue.push(_functors[i].get());
        }
    }

    // TODO: concurrent execution for all ready nodes
    while (!queue.empty())
    {
        FunctorNode *node = queue.front();
        queue.pop();

        assert(status[node->index] == 0);
        func(*node);
        status[node->index] = -1; // visited
        for (auto down_node : node->down_functors)
        {
            --status[down_node->index]; // current node has executed
            if (status[down_node->index] == 0)
            {
                queue.push(down_node.get());
            }
        }
    }

    assert(status.bool_func([](int e) { return e == -1; }));
}

/*
void backward(const TensorD<double> &weights)
{
    // first, use each value's weight as their grad
    assert(weights.size() == size());
    this->grad().copy(weights, 0, weights.size(), dim());

    // second, paint all the functors/tensors that reversely depend on this tensor;
    _g.prune(*this);

    // third, start to run the whole graph in reverse order
    _g.backward(*this);
}*/

void FunctorGraph::backward(TensorList &_, const TensorList &y)
{
    assert(y.size() >= 1);
    backward(y);
}

// start_tensor's grad are set
void FunctorGraph::backward(const TensorList& output_tensor_grads)
{
    // TODO: check whether this start_tensor is in the graph
    // TODO: prune the graph to cut all useless nodes that not useful for loss
    // TODO: before that, we only support in the graph all the nodes are useful
    //this->prune(start_tensor);

    // copy grads
    assert(output_tensor_grads.size() == _output_tensors.size());
    for (uint i = 0; i < output_tensor_grads.size(); ++i)
    {
        _output_tensors[i].grad().weak_copy(output_tensor_grads[i].grad());
    }

    backward_traverse([](FunctorNode &node) -> void {
        /*const TensorList *node_x;
        const TensorList *node_y;
        TensorList *node_x_grad;
        if (node.use_input)
        {
            node_x = &x;
            node_y = &y;
            node_x_grad = &x_grad;
        }
        else
        {
            node_x = &node.inputs;
            node_y = &node.outputs;
            node_x_grad = &node.input_grads;
        }

        // output_grads sum up by down nodes' x_grad
        // end node(e.g., loss node) will skip this
        // TODO: what if end node is not loss node?? we need to disable the calculations for all such nodes in backpropagation
        for (uint i = 0; i < node.output_grads.size(); ++i)
        {
            node.output_grads[i]->init(node.outputs[i]->dim());

            if has output_grads_linked_grads
            for (auto linked_grad : node.output_grad_linked_grads[i])
            {
                node.output_grads[i]->add(*linked_grad);
            }
            for (auto down_node : node.output_nodes)
            {
                for (uint j = 0; j < down_node->input_nodes.size(); ++j)
                {
                    if (down_node->input_nodes[j].first == &node && down_node->input_nodes[j].second == i)
                    {
                        node.output_grads[i]->add_(*down_node->input_grads[j]);
                        // TODO: usually just one match per inner loop, and then no need the single tensor copy
                    }
                }
            }
        }*/

        PFunctor func = node.func;
        // LOG_INFO("Backward functor:" << func->type());
        func->backward(node.inputs, node.outputs);
    });
}

void FunctorGraph::backward_traverse(const std::function<void(FunctorNode &)> &func) const
{
    // TODO: we need to ensure this functor's output grads are all ready
    // TODO: if it's not ready, there may be a lot of nodes have no grad calculated
    Vector<int> status(_functors.size(), TensorInit_Types::One);
    //PFunctorNode start_functor = start_tensor->output_functor;
    //status[start_functor->index] = 0;

    std::queue<FunctorNode *> queue;
    //queue.push(start_functor.get());

    // note: collect all starting functors
    // TODO: it may happen start_tensor is NOT end of graph, and then we need disable the tree under start_tensor
    for (uint i = 0; i < _functors.size(); ++i)
    {
        status[i] = _functors[i]->down_functors.size();
        if (status[i] == 0)
        {
            queue.push(_functors[i].get());
        }
    }

    // TODO: concurrent execution for all ready nodes
    while (!queue.empty())
    {
        FunctorNode *node = queue.front();
        queue.pop();

        assert(status[node->index] == 0);
        func(*node);
        status[node->index] = -1; // visited
        for (auto up_node : node->up_functors)
        {
            --status[up_node->index];
            if (status[up_node->index] == 0)
            {
                queue.push(up_node.get());
            }
        }
    }

    assert(status.bool_func([](int e) { return e == -1; }));
}

void FunctorGraph::prune(const Tensor& start_tensor)
{
}