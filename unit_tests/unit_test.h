#pragma once
#include "../inc/0.tensors/predefs.h"
#include "../inc/0.tensors/string_util.h"

class TestClass
{
public:
    virtual std::pair<Array<Func_Void>, Array<std::string>> get_test_cases() = 0;
};

#define REGISTER_TEST_CASES(cases...) OVERRIDE std::pair<Array<Func_Void>, Array<std::string>> get_test_cases() {return std::make_pair(Array<Func_Void>{cases}, StringUtil::split(#cases, ","));}
#define REGISTER_TEST_CLASS(c) std::make_pair(#c, std::make_shared<c>())