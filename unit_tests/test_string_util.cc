#pragma once
#include "unit_test.h"
#include "../inc/0.tensors/string_util.h"

class TestStringUtil: public TestClass
{
public:
    REGISTER_TEST_CASES(test_trim, test_split)

    static void test_trim()
    {
        std::string s = "    a b  c  ";
        assert(StringUtil::trim(s) == "a b  c");
    }

    static void test_split()
    {
        assert(StringUtil::split("a b  c", " ").size() == 3);
    }
};