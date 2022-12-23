#pragma once
#include "array.h"
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
};