#pragma once

#include "operator_base.h"

// variant projection op: tensor[N, L, I] -> tensor[N, L, O]
// TODO: supports one input
// TODO: supports one output
// Note: seems the next iteration of forward, hidden tensor is reset, could we reuse last iteration's?
// Note: Rnn supports auto_grad now
// Note: not supporting save/load
class RnnBase : public Operator
{
protected:
    bool _hidden_as_output;
    uint _input_dim;
    uint _hidden_dim;
    uint _output_dim;
    bool _has_bias;
    TensorInit_Types _init_type;
    Activation_Types _hidden_activation;
    Activation_Types _output_activation;

    RnnBase(const std::string& type, uint input_dim, uint hidden_dim, uint output_dim, bool has_bias = true, bool hidden_as_output = false, TensorInit_Types init_type = TensorInit_Types::Gaussian,
            Activation_Types hidden_activation = Activation_Types::Tanh, Activation_Types output_activation = Activation_Types::Sigmoid)
        : Operator(nullptr, type), _hidden_as_output(hidden_as_output), _input_dim(input_dim), _hidden_dim(hidden_dim), _output_dim(output_dim), _has_bias(has_bias), _init_type(init_type),
          _hidden_activation(hidden_activation), _output_activation(output_activation)
    {
        assert(input_dim > 0 && hidden_dim > 0 && output_dim > 0);
    }

protected:
    // h = act(h*u + x*w + b)
    static Tensor Hidden_Unit(const Tensor& input, const Tensor& hidden,
                            const Tensor&w, const Tensor&u, const Tensor& b, Activation_Types act)
    {
        // h:[batch_size, hidden_dim] dot u:[hidden_dim, hidden_dim] -> hidden_new:[batch_size, hidden_dim]
        Tensor hidden_new = hidden.dot(u);
        // x:[node_len, batch_size, input_dim] dot w:[hidden_dim, input_dim] -> out:[batch_size, hidden_dim]
        // input.dot(w, *out, 0, -1, true, node_id); TODO: this is the only place used t1_id, but now i used divide/combine to resolve this
        hidden_new.add_(input.dot(w)); // note: add_to is not working in the whole system
        if (b != nullptr && b.size() == hidden_new.size())
        {
            hidden_new.add_(b); // x * _w + h * _u + b1
        }

        hidden_new.activation_(act);
        return hidden_new;
    }

    virtual void process_node(const Tensor& input, const Tensor& hidden, Tensor& output_new, Tensor& hidden_new) const = 0;

public:
    // hidden: [batch_size, hidden_dim]
    // out: [batch_size, node_len, output_dim]
    virtual uint output_tensor_count() const override
    {
        return 2;
    }

    // x[0]/input: [batch_size, node_len, input_dim]
    // y[0]/hidden: [batch_size, hidden_dim]
    // y[1]/out: [batch_size, node_len, output_dim]
    virtual TensorList forward(const TensorList &x) const override
    {
        // preprocess x
        assert(x.size() >= 1);
        Tensor input = x[0];
        assert(input.shape() == 3); // [batch_size, node_len, input_dim]
        auto input_swap = input.swap(0, 1); // [node_len, batch_size, input_dim]
        uint node_len = input_swap.dim()[0];
        uint batch_size = input_swap.dim()[1];
        assert(_input_dim == input_swap.dim()[2]);
        auto input_per_node = input_swap.divide(1); // [batch_size, input_dim]

        // prepare y
        TensorList y;
        y.reserve(2);
        Tensor hidden = y[0], output = y[1];
        hidden.reset({batch_size, _hidden_dim}, TensorInit_Types::Zero); // initial hidden is set to 0
        output.reset({node_len, batch_size, _output_dim});     // [node_len, batch_size, output_dim]
        TensorList output_per_node;

        Tensor hidden_new;
        Tensor output_new({batch_size, _output_dim}, TensorInit_Types::Zero);
        for (uint i = 0; i < input_per_node.size(); ++i)
        {
            process_node(input_per_node[i], hidden, output_new, hidden_new);
            if (_hidden_as_output)
            {
                assert (_hidden_dim == _output_dim);
                output_per_node.push_back(hidden_new);
            }
            else
            {
                output_per_node.push_back(output_new);
            }
            
            hidden = hidden_new;
        }

        y[1] = Tensor::combine(output_per_node); // [node_len, batch_size, output_dim]
        y[1] = y[1].swap(0, 1);

        return y;
    }
};

class RawRnn : public RnnBase
{
private:
    Tensor _w, _u, _v, _b1, _b2;

public:
    RawRnn(uint input_dim, uint hidden_dim, uint output_dim, bool has_bias = true, TensorInit_Types init_type = TensorInit_Types::Gaussian,
        Activation_Types hidden_activation = Activation_Types::Tanh, Activation_Types output_activation = Activation_Types::Sigmoid)
        : RnnBase("RawRnn", input_dim, hidden_dim, output_dim, has_bias, false, init_type, hidden_activation, output_activation)
    {
        _w = add_param("w", {hidden_dim, input_dim}, init_type);
        _u = add_param("u", {hidden_dim, hidden_dim}, init_type);
        _v = add_param("v", {output_dim, hidden_dim}, init_type);

        if (_has_bias)
        {
            _b1 = add_param("b1", {hidden_dim}, init_type);
            _b2 = add_param("b2", {output_dim}, init_type);
        }
    }

public:
    // h = tanh(x * _w + h * _u + b1)
    // out = sigmoid(h * _v + b2)
    virtual void process_node(const Tensor& input, const Tensor& hidden, Tensor& output_new, Tensor& hidden_new) const override
    {
        hidden_new = Hidden_Unit(input, hidden, _w, _u, _b1, _hidden_activation);

        output_new = hidden_new.dot(_v); 
        if (_has_bias)
        {
            output_new.add_(_b2);
        }

        output_new.activation_(_output_activation); // [batch_size, output_dim]
    }
};

class Lstm : public RnnBase
{
private:
    Tensor _wf, _wi, _wo, _wc, _uf, _ui, _uo, _uc, _bf, _bi, _bo, _bc;

public:
    Lstm(uint input_dim, uint hidden_dim, bool has_bias = true, TensorInit_Types init_type = TensorInit_Types::Gaussian,
         Activation_Types hidden_activation = Activation_Types::Tanh, Activation_Types output_activation = Activation_Types::Sigmoid)
        : RnnBase("Lstm", input_dim, hidden_dim, hidden_dim, has_bias, true, init_type, hidden_activation, output_activation)
    {
        _wf = add_param("wf", {hidden_dim, input_dim}, init_type);
        _wi = add_param("wi", {hidden_dim, input_dim}, init_type);
        _wo = add_param("wo", {hidden_dim, input_dim}, init_type);
        _wc = add_param("wc", {hidden_dim, input_dim}, init_type);

        _uf = add_param("uf", {hidden_dim, hidden_dim}, init_type);
        _ui = add_param("ui", {hidden_dim, hidden_dim}, init_type);
        _uo = add_param("uo", {hidden_dim, hidden_dim}, init_type);
        _uc = add_param("uc", {hidden_dim, hidden_dim}, init_type);

        if (_has_bias)
        {
            _bf = add_param("bf", {hidden_dim}, init_type);
            _bi = add_param("bi", {hidden_dim}, init_type);
            _bo = add_param("bo", {hidden_dim}, init_type);
            _bc = add_param("bc", {hidden_dim}, init_type);
        }
    }

public:
    // x: [batch_size, node_len, input_dim]
    // y[0]: [batch_size, hidden_dim]
    // y[1]: [batch_size, node_len, hidden_dim]
    // h: [batch_size, hidden_dim]
    // f = sigmoid(x * _wf + h * _uf + bf)
    // i = sigmoid(x * _wi + h * _ui + bi)
    // o = sigmoid(x * _wo + h * _uo + bo)
    // c = sigmoid(x * _wc + h * _uc + bc)
    // out = f mul out + i mul c
    // h = o mul sigmoid(out)
    virtual void process_node(const Tensor& input, const Tensor& hidden, Tensor& output_new, Tensor& hidden_new) const override
    {
        auto f = RnnBase::Hidden_Unit(input, hidden, _wf, _uf, _bf, _hidden_activation);
        auto i = RnnBase::Hidden_Unit(input, hidden, _wi, _ui, _bi, _hidden_activation);
        auto o = RnnBase::Hidden_Unit(input, hidden, _wo, _uo, _bo, _hidden_activation);
        auto c = RnnBase::Hidden_Unit(input, hidden, _wc, _uc, _bc, _hidden_activation);

        //note out_new is used to as c_plus here
        output_new.mul_(f).add_(c.mul(i)); // note: could use add_to

        hidden_new = output_new.activation(_hidden_activation).mul_(o);

    }
};

class Gru : public RnnBase
{
private:
    Tensor _wz, _wr, _wh, _uz, _ur, _uh, _bz, _br, _bh;

public:
    Gru(uint input_dim, uint hidden_dim, bool has_bias = true, TensorInit_Types init_type = TensorInit_Types::Gaussian,
        Activation_Types hidden_activation = Activation_Types::Tanh, Activation_Types output_activation = Activation_Types::Sigmoid)
        : RnnBase("Gru", input_dim, hidden_dim, hidden_dim, has_bias, true, init_type, hidden_activation, output_activation)
    {
        _wz = add_param("wz", {hidden_dim, input_dim}, init_type);
        _wr = add_param("wr", {hidden_dim, input_dim}, init_type);
        _wh = add_param("wh", {hidden_dim, input_dim}, init_type);

        _uz = add_param("uz", {hidden_dim, hidden_dim}, init_type);
        _ur = add_param("ur", {hidden_dim, hidden_dim}, init_type);
        _uh = add_param("uh", {hidden_dim, hidden_dim}, init_type);

        if (_has_bias)
        {
            _bz = add_param("bz", {hidden_dim}, init_type);
            _br = add_param("br", {hidden_dim}, init_type);
            _bh = add_param("bh", {hidden_dim}, init_type);
        }
    }

public:
    // x: [batch_size, node_len, input_dim]
    // y[0]: [batch_size, hidden_dim]
    // y[1]: [batch_size, node_len, hidden_dim]
    // h: [batch_size, hidden_dim]
    // z = sigmoid(x * _wz + h * _uz + bz)
    // r = sigmoid(x * _wr + h * _ur + br)
    // h_origin = sigmoid(x * _wh + h * _uh + bh)
    // h = h mul (1 - z) + h_origin mul z
    virtual void process_node(const Tensor& input, const Tensor& hidden, Tensor& output_new, Tensor& hidden_new) const override
    {
        auto z = RnnBase::Hidden_Unit(input, hidden, _wz, _uz, _bz, _hidden_activation);
        auto r = RnnBase::Hidden_Unit(input, hidden, _wr, _ur, _br, _hidden_activation);
        hidden_new = RnnBase::Hidden_Unit(input, hidden, _wh, _uh, _bh, _hidden_activation);
        hidden_new.mul_(z).add_(hidden.mul(z.linear(-1.0, 1)));
    }
};