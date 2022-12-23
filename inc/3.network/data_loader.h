#pragma once

#include "../1.functors/tensor_node.h"

class DataLoaderConfig
{
};

class DataLoadersManagerConfig
{
};

class DataLoader
{
public:
    // return how many samples readed
    virtual uint read_next(Tensor &data, uint mini_batch) = 0;

    // return to the beginning of the content, may read from cache
    virtual void reset() = 0;
};

class InMemoryDataLoader : public DataLoader
{
private:
    Tensor _x;
    uint _offset;
public:
    InMemoryDataLoader(const Tensor& x) : _x(x), _offset(0)
    {
    }

    virtual uint read_next(Tensor &data, uint mini_batch) override
    {
        assert(_x.shape() > 0);
        if (_offset >= _x.dim()[0])
        {
            return 0;
        }
        else if (_offset + mini_batch >= _x.dim()[0])
        {
            mini_batch = _x.dim()[0] - _offset;
        }

        Vector<uint> dim(_x.dim());
        dim[0] = mini_batch;
        _x.set_auto_grad(false);
        data = _x.subset(dim, _offset * Vector(dim.subset(1)).product());
        _offset += mini_batch;
        return mini_batch;
    }

    virtual void reset() override
    {
        this->_offset = 0;
    }
};

class DataLoadersManager : public Array<Ptr<DataLoader>>
{
public:
    static bool parse_config(const std::string &config_file, DataLoadersManagerConfig &config)
    {
        // TODO
        assert(false);
        return false;
    }

    bool init(const std::string &config_file)
    {
        DataLoadersManagerConfig config;
        if (!parse_config(config_file, config))
        {
            LOG_ERROR("data loaders manager parse config failed");
            return false;
        }

        init(config);
        return true;
    }

    bool init(const DataLoadersManagerConfig &config)
    {
        // TODO
        assert(false);
        return false;
    }
};