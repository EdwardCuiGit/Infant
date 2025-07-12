#pragma once
#include "vector.h"
#include <cstring>

class StringUtil
{
public:
    static std::string &ltrim(std::string &str, const std::string &chars = "\t\n\v\f\r ")
    {
        str.erase(0, str.find_first_not_of(chars));
        return str;
    }

    static std::string &rtrim(std::string &str, const std::string &chars = "\t\n\v\f\r ")
    {
        str.erase(str.find_last_not_of(chars) + 1);
        return str;
    }

    static std::string &trim(std::string &str, const std::string &chars = "\t\n\v\f\r ")
    {
        return ltrim(rtrim(str, chars), chars);
    }

    static Array<std::string> split(const std::string &str, const std::string &sep)
    {
        Array<std::string> res;
        if ("" == str)
            return res;
        char *strs = new char[str.length() + 1];
        std::strcpy(strs, str.c_str());

        char *seps = new char[sep.length() + 1];
        std::strcpy(seps, sep.c_str());

        char *p = std::strtok(strs, seps);
        while (p)
        {
            std::string s = p;
            res.push_back(StringUtil::trim(s));
            p = strtok(NULL, seps);
        }

        delete[] strs;
        delete[] seps;
        return res;
    }

    static std::string read_line(std::istream& i)
    {
        std::string line;
        std::getline(i, line);
        return line;
    }

    // prefix = 123
    static uint read_uint(std::istream& i, const std::string& prefix, const std::string& sep = "=")
    {
        std::string line = read_line(i);
        auto fields = split(line, sep);
        assert(fields.size() == 2);
        assert(prefix == fields[0]);
        int value = std::stoi(fields[1]);
        assert(value >= 0);
        return (uint)value;
    }

    static void write_uint(std::ostream& o, const std::string& prefix, uint value, const std::string& sep = "=")
    {
        o << prefix << " " << sep << " " << value << "\n";
    }

    // 0 2 1
    static void read_uint_vector(std::istream& i, Vector<uint>& output, const std::string& sep = " ")
    {
        std::string line = read_line(i);
        auto fields = split(line, sep);
        // can't exceed allocated size, shall we truncate?
        output.reserve(fields.size());
        for(uint j = 0; j < fields.size(); ++j)
        {
            int v = std::stoi(fields[j]);
            assert(v >= 0);
            output[j] = (uint)v;
        }
    }

    template <class T>
    static void write_vector(std::ostream& o, const Vector<T>& input, const std::string& sep = " ")
    {
        for(uint i = 0; i < input.size() - 1; ++i)
        {
            o << input[i] << sep;
        }

        o << input[input.size() - 1] << "\n";
    }

    // 0.1 0.2 0.1
    static void read_float_vector(std::istream& i, Vector<float>& output, const std::string& sep = " ")
    {
        std::string line = read_line(i);
        auto fields = split(line, sep);
        // can't exceed allocated size, shall we truncate?
        assert(output.size() >= fields.size());
        for(uint i = 0; i < fields.size(); ++i)
        {
            output[i] = std::stod(fields[i]);
        }
    }

    // prefix = a b c
    static std::string read_string(std::istream& i, const std::string& prefix, const std::string& sep = "=")
    {
        std::string line = read_line(i);
        auto fields = split(line, sep);
        assert(fields.size() == 2);
        assert(prefix == fields[0]);
        return fields[1];
    }

    static void write_string(std::ostream& o, const std::string& prefix, const std::string& value, const std::string& sep = "=")
    {
        o << prefix << " " << sep << " " << value << "\n";
    }

    static void assert_next_line(std::istream& i, const std::string& expected)
    {
        std::string line;
        std::getline(i, line);
        assert(line == expected);
    }

    static Str concat(const std::string& prefix, uint id)
    {
        std::stringstream str_stream;
        str_stream << prefix;
        str_stream << id;
        return str_stream.str();
    }
};