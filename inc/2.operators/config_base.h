#pragma once
#include "../0.tensors/vector.h"
#include "../0.tensors/string_util.h"

struct ConfigBase
{
    friend class TestConfigBase;

// not support Vector<uint>
#define DEFINE_FIELD(type, name, default_value)                  \
    type &name() { return access_##type(#name, default_value); } \
    type name() const { return access_##type(#name); }
    // uint& output_dim(){ return access_uint("output_dim");}

#define DEFINE_SUB_CONFIG(type, name)                                                  \
    inline type::Config &name()                                                        \
    {                                                                                  \
        return *(std::static_pointer_cast<type::Config>(access_config(#name, #type))); \
    }                                                                                  \
                                                                                       \
    inline const type::Config &name() const                                            \
    {                                                                                  \
        return *(std::static_pointer_cast<type::Config>(access_config(#name, #type))); \
    }

private:
    std::string _type;
    Map<const std::string, uint> _m_uint;
    Map<const std::string, int> _m_int;
    Map<const std::string, bool> _m_bool;
    Map<const std::string, float> _m_float;
    Map<const std::string, Vector<uint>> _m_uint_vector;
    // name, => {type, ConfigBase*}
    Map<const std::string, Ptr<ConfigBase>> _m_config;

    using Operator_Config_Func = Ptr<ConfigBase> (*)();
    friend class Operator; // config classes will be registered in the start of thread
    static Map<std::string, Operator_Config_Func> _Op_Config_Factory;

public:
    ConfigBase(const std::string& type)
    {
        _type = type;
    }

    inline Ptr<ConfigBase> &access_config(const std::string &name, const std::string &type)
    {
        auto res = this->_m_config.find(name);
        if (res == this->_m_config.end())
        {
            auto config_ptr = Create_Config(type);
            this->_m_config.emplace(name, config_ptr);
        }

        return this->_m_config[name];
    }

    inline uint &access_uint(const std::string &name, uint default_value = 0)
    {
        auto res = this->_m_uint.find(name);
        if (res == this->_m_uint.end())
        {
            this->_m_uint[name] = default_value;
        }

        return this->_m_uint[name];
    }

    inline int &access_int(const std::string &name, int default_value = 0)
    {
        auto res = this->_m_int.find(name);
        if (res == this->_m_int.end())
        {
            this->_m_int[name] = default_value;
        }

        return this->_m_int[name];
    }

    inline bool &access_bool(const std::string &name, bool default_value = false)
    {
        auto res = this->_m_bool.find(name);
        if (res == this->_m_bool.end())
        {
            this->_m_bool[name] = default_value;
        }

        return this->_m_bool[name];
    }

    inline float &access_float(const std::string &name, float default_value = 0)
    {
        auto res = this->_m_float.find(name);
        if (res == this->_m_float.end())
        {
            this->_m_float[name] = default_value;
        }

        return this->_m_float[name];
    }

    inline Vector<uint> &access_uint_vector(const std::string &name)
    {
        auto res = this->_m_uint_vector.find(name);
        if (res == this->_m_uint_vector.end())
        {
            this->_m_uint_vector[name] = Vector<uint>();
        }

        return this->_m_uint_vector[name];
    }

    inline const Ptr<ConfigBase> &access_config(const std::string &name, const std::string &type) const
    {
        auto res = this->_m_config.find(name);
        if (res == this->_m_config.end())
        {
            assert(false);
        }

        return res->second;
    }

    inline uint access_uint(const std::string &name) const
    {
        auto res = this->_m_uint.find(name);
        if (res == this->_m_uint.end())
        {
            assert(false);
        }

        return res->second;
    }

    inline int access_int(const std::string &name) const
    {
        auto res = this->_m_int.find(name);
        if (res == this->_m_int.end())
        {
            assert(false);
        }

        return res->second;
    }

    inline bool access_bool(const std::string &name) const
    {
        auto res = this->_m_bool.find(name);
        if (res == this->_m_bool.end())
        {
            assert(false);
        }

        return res->second;
    }

    inline float access_float(const std::string &name) const
    {
        auto res = this->_m_float.find(name);
        if (res == this->_m_float.end())
        {
            assert(false);
        }

        return res->second;
    }

    inline Vector<uint> access_uint_vector(const std::string &name) const
    {
        auto res = this->_m_uint_vector.find(name);
        if (res == this->_m_uint_vector.end())
        {
            assert(false);
        }

        return res->second;
    }

public:
    static Ptr<ConfigBase> Create_Config(const std::string type)
    {
        auto op_config_func = ConfigBase::_Op_Config_Factory.find(type);
        assert(op_config_func != ConfigBase::_Op_Config_Factory.end());
        auto config_ptr = op_config_func->second();
        config_ptr->_type = type;
        return config_ptr;
    }

    virtual void save(std::ostream &o)
    {
        o << "start of config\n";
        for (auto pair : _m_uint)
        {
            o << pair.first << " " << (uint)NumTypes::Uint << " " << pair.second << "\n";
        }

        for (auto pair : _m_int)
        {
            o << pair.first << " " << (uint)NumTypes::Int << " " << pair.second << "\n";
        }

        for (auto pair : _m_bool)
        {
            o << pair.first << " " << (uint)NumTypes::Bool << " " << pair.second << "\n";
        }

        for (auto pair : _m_float)
        {
            o << pair.first << " " << (uint)NumTypes::Double << " " << pair.second << "\n";
        }

        for (auto pair : _m_uint_vector)
        {
            o << pair.first << " " << (uint)NumTypes::UintVector << " ";

            for (uint j = 0; j < pair.second.size(); ++j)
            {
                o << pair.second[j] << " ";
            }

            o << "\n";
        }

        for (auto pair : _m_config)
        {
            o << pair.first << " " << (uint)NumTypes::SubConfig << " " << pair.second->_type << "\n";
            pair.second->save(o);
        }

        o << "end of config\n";
    }

    virtual void load(std::istream &i)
    {
        StringUtil::assert_next_line(i, "start of config");
        std::string line = StringUtil::read_line(i);
        while (line != "end of config")
        {
            auto fields = StringUtil::split(line, " ");
            assert(fields.size() >= 2);
            std::string name = fields[0];
            int num_type = std::atoi(fields[1].c_str());
            switch (num_type)
            {
            case (int)NumTypes::Uint:
                assert(fields.size() == 3);
                access_uint(name) = std::stoi(fields[2]);
                break;
            case (int)NumTypes::Int:
                assert(fields.size() == 3);
                access_int(name) = std::stoi(fields[2]);
                break;

            case (int)NumTypes::Bool:
                assert(fields.size() == 3);
                access_bool(name) = std::stoi(fields[2]) == 1;
                break;
            case (int)NumTypes::Double:
                assert(fields.size() == 3);
                access_float(name) = std::stod(fields[2]);
                break;
            case (int)NumTypes::UintVector:
                for (uint j = 2; j < fields.size(); ++j)
                {
                    access_uint_vector(name).push_back(std::stoi(fields[j]));
                }
                break;
            case (int)NumTypes::SubConfig:
            {
                assert(fields.size() == 3);
                Str config_type = fields[2];
                access_config(name, config_type)->load(i);
                break;
            }
            default:
                assert(false);
                break;
            }

            line = StringUtil::read_line(i);
        }
    }
};

// this is used for additional parameters at runtime
// below constructors are for quick one param config, if you need several add one by one
class RTConfig : public ConfigBase
{
public:
    RTConfig() : ConfigBase("RTConfig")
    {
    }

    RTConfig(const Str& name, bool value = false) : RTConfig()
    {
        access_bool(name) = value;
    }

    RTConfig(const Str& name, uint value = 0) : RTConfig()
    {
        access_uint(name) = value;
    }

    RTConfig(const Str& name, float value = 0.0) : RTConfig()
    {
        access_float(name) = value;
    }

    RTConfig(const Str& name, int value = 0) : RTConfig()
    {
        access_int(name) = value;
    }
};
