#pragma once

#include "inc/1.functors/tensor_node.h"
#include "config_base.h"

// Learnable Operator: added parameters
// TODO: add required shape for both input & outputs
class Operator : public Functor
{
#define REGISTER_OP(name)                                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                                     \
        Operator::Register_Op(#name, [](const ConfigBase &c) -> Ptr<Operator> { return std::static_pointer_cast<Operator>(std::make_shared<name>(dynamic_cast<const name::Config &>(c))); }, []() -> Ptr<ConfigBase> { return std::static_pointer_cast<ConfigBase>(std::make_shared<name::Config>()); }); \
    }
    //{"Fc", []()->Ptr<Operator>{ return std::static_pointer_cast<Operator>(std::make_shared<Fc>());}}

protected:
    // below are used for operator registration
    using Operator_Func = Ptr<Operator> (*)(const ConfigBase &);
    static Map<std::string, Operator_Func> _Op_Factory;

    // parameter list
    mutable Map<std::string, Tensor> _p;

    // config pointer
    ConfigBase *_base_config_ptr = nullptr;

    // sub operators
    Map<std::string, Ptr<Operator>> _o;

public:
    // used for op registration during env init
    static void Register_Op(const Str &name, Operator_Func func, ConfigBase::Operator_Config_Func config_func)
    {
        _Op_Factory[name] = func;
        ConfigBase::_Op_Config_Factory[name] = config_func;
    }

    static Ptr<Operator> Load_Op(std::istream &i)
    {
        Str op_type = StringUtil::read_string(i, "Operator Type");
        auto op_id = StringUtil::read_string(i, "Operator Id");

        // load config of op
        auto op_config = ConfigBase::Create_Config(op_type);
        op_config->load(i);

        // create op
        auto op_func = _Op_Factory.find(op_type);
        assert(op_func != _Op_Factory.end());
        Ptr<Operator> op = op_func->second(*op_config);
        op->_type = op_type;

        op->_id = op_id;

        // load parameters of op
        // TODO: didn't validate configs are consistent with parameter dims
        op->_load_params(i);

        return op;
    }

    void save_op(std::ostream &o) const
    {
        StringUtil::write_string(o, "Operator Type", this->type());
        StringUtil::write_string(o, "Operator Id", this->_id);
        if (_base_config_ptr != nullptr)
            _base_config_ptr->save(o);

        this->_save_params(o);
    }

    virtual TensorList forward(const TensorList &x, const RTConfig& rc) const
    {
        assert(false);
        return x;
    }

    virtual TensorList forward(const TensorList &x) const
    {
        assert(false);
        return x;
    }

    virtual bool is_const() const override
    {
        for (auto o : _o)
        {
            if (!o.second->is_const())
            {
                return false;
            }
        }

        return true;
    }

    uint parameter_count() const
    {
        uint total = 0;
        for (auto p : _p)
        {
            total += p.second.data().size();
        }

        return total;
    }

    auto parameters()
    {
        return _p;
    }

    /*
    virtual bool has_loss() const
    {
        return false;
    }*/

protected:
    Operator(ConfigBase *ptr, const std::string &type = "") : Functor(type), _base_config_ptr(ptr)
    {
        if (ptr != nullptr)
        {
            // ptr->_type = type;
            // note: this is for auto Config type set, but base class constructor can't access sub class fields as it's not 
            // generated yet
        }
        else
        {
            LOG_WARN("ConfigBase is nullptr, which means can not auto save and load:" << type);
        }
    }

    template <typename _Tp, typename _ConfigType>
    Ptr<_Tp> add_op(const std::string &id, const _ConfigType &c)
    {
        Ptr<_Tp> op = std::make_shared<_Tp>(c);
        op->_id = id;
        op->_type = c._type;
        assert(_o.find(id) == _o.end());
        _o[id] = op;
        return op;
    }

    inline Tensor add_param(const std::string &id, const Vector<uint> &dims, TensorInit_Types init_type)
    {
        Tensor p(dims, init_type, id, true);
        assert(_p.find(id) == _p.end());
        _p[id] = p;
        return p;
    }

protected:
    virtual void forward(const TensorList &x, TensorList &y) const override
    {
        y = forward(x);
        assert(false);
    }

    // don't use this func, instead operator() will be used as non const func
    // auto_grad, not used below, pls do not override this
    SEALED virtual void backward(TensorList &x, const TensorList &y) override
    {
        assert(false);
    }

private:
    inline void _save_params(std::ostream &o) const
    {
        o << "start of params of one operator\n";
        StringUtil::write_uint(o, "param_size", _p.size());
        for (auto pair : _p)
        {
            pair.second.save(o);
        }

        // the whole op graph has created, but parameters are not saved yet
        StringUtil::write_uint(o, "Sub_Operator_Count", _o.size());
        for (auto pair : _o)
        {
            StringUtil::write_string(o, "Sub_Op_Id", pair.first);
            pair.second->_save_params(o);
        }

        o << "end of params of one operator\n";
    }

    inline void _load_params(std::istream &i)
    {
        StringUtil::assert_next_line(i, "start of params of one operator");
        uint param_size = StringUtil::read_uint(i, "param_size");
        assert(param_size > 0);

        for (uint j = 0; j < param_size; ++j)
        {
            Tensor t;
            t.load(i);
            _p[t.id()] = t;
        }

        // the whole op graph has created, but parameters are not loaded yet
        uint sub_op_count = StringUtil::read_uint(i, "Sub_Operator_Count");
        for (uint j = 0; j < sub_op_count; ++j)
        {
            auto sub_op_id = StringUtil::read_string(i, "Sub_Op_Id");
            auto sub_op = this->_o.find(sub_op_id);
            assert(sub_op != this->_o.end());
            sub_op->second->_load_params(i);
        }

        StringUtil::assert_next_line(i, "end of params of one operator");
    }
};

// Unary Learnable Operator
class UnOp : public Operator
{
protected:
    UnOp(ConfigBase *ptr, const std::string &type) : Operator(ptr, type) {}
public:
    virtual Tensor forward(const Tensor &x) const = 0;
protected:
    SEALED virtual TensorList forward(const TensorList &x) const override
    {
        assert(x.size() >= 1);
        Tensor y = forward(x[0]);
        return {y};
    }
};
