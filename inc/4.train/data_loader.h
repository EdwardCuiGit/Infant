#pragma once

#include "inc/1.functors/tensor_node.h"
#include <iostream>
#include <fstream>

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
    static bool Parse_config(const std::string &config_file, DataLoadersManagerConfig &config)
    {
        // TODO
        assert(false);
        return false;
    }

    bool init(const std::string &config_file)
    {
        DataLoadersManagerConfig config;
        if (!Parse_config(config_file, config))
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

class TextFileLoader : public DataLoader
{
private:
    std::ifstream _f;
    std::string _file_path;
    uint _max_node_len;
public:
    TextFileLoader(const std::string& file_path, uint max_node_len = 1) : _file_path(file_path), _max_node_len(max_node_len)
    {
    }

    ~TextFileLoader()
    {
        _f.close();
    }

    // return how many samples readed
    virtual uint read_next(Tensor &data, uint mini_batch) override
    {
        if (!_f.is_open())
        {
            _f.open(_file_path);
            assert(_f.is_open());
        }

        std::string line_txt;
        Array<std::string> lines_txt;
        uint rows = 0;
        while (rows < mini_batch && std::getline(_f, line_txt))
        {
            if (line_txt.size() > 0)
                lines_txt.push_back(line_txt);
            rows++;
        }

        this->_text_to_tensor(lines_txt, data);
        return rows;
    }

    /*
    special words:
    dict_id == 0: <pad>
    dict_id == 1: <start>
    dict_id == 2: <end>
    */
    virtual Tensor _text_to_tensor(const Array<std::string>& lines_txt, Tensor& data) const
    {
        // by default, it's convert each char to be it's ascii value
        Vector<float> _vector(lines_txt.size() * _max_node_len);
        for (uint row = 0; row < lines_txt.size(); ++row)
        {
            const char* ch = lines_txt[row].data();
            bool found_end = false;
            for (uint node_count = 0; node_count < _max_node_len; node_count++)
            {
                float v;
                if (*ch != 0)
                {
                    if (node_count == _max_node_len - 1)
                    {
                        v = 2; // add <end> to the end of the line
                        found_end = true;
                    }
                    else
                    {
                        // v = *ch;
                        if (*ch >= '0' && *ch <= '9')
                            v = *ch - '0' + 10;
                        else if (*ch == '+')
                            v = 3;
                        else if (*ch == '-')
                            v = 4;
                        else if (*ch == '*')
                            v = 5;
                        else if (*ch == '/')
                            v = 6;
                        else
                            v = 7;
                        ch++;
                    }
                } 
                else
                {
                    if (!found_end)
                    {
                        v = 2; // add <end> to the end of the line
                        found_end = true;
                    }
                    else
                    {
                        v = 0; // padding
                    }
                }

                _vector[row * _max_node_len + node_count] = v;
            }
        }

        // note: 3x memory for each min_batch, we shall optimize it
        if (lines_txt.size() > 0)
        {
            data.reset({lines_txt.size(), _max_node_len}, _vector);
            return data;
        }
        else
        {
            return Tensor();
        }
    }

    // return to the beginning of the content, may read from cache
    virtual void reset() override
    {
        _f.close();
    }
};
