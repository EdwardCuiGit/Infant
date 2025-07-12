#pragma once
#include "unit_test.h"
#include "../inc/0.tensors/string_util.h"

class TestStringUtil: public TestClass
{
public:
    REGISTER_TEST_CASES(test_trim, test_split, test_read_and_write_uint, test_read_and_write_string, test_assert_next_line,
    test_read_and_write_float_vector, test_read_and_write_uint_vector)

    static void test_trim()
    {
        std::string s = "    a b  c  ";
        assert(StringUtil::trim(s) == "a b  c");
    }

    static void test_split()
    {
        assert(StringUtil::split("a b  c", " ").size() == 3);
    }

    static void test_read_and_write_uint()
    {
        std::stringstream i, o;
        i << "input_dim = " << 234 << "\n";
        assert(StringUtil::read_uint(i, "input_dim") == 234);

        StringUtil::write_uint(o, "xxx", 222);
        assert(o.str() == "xxx = 222\n");
    }

    static void test_read_and_write_string()
    {
        std::stringstream i, o;
        i << "input_dim = " << "xxx" << "\n";
        assert(StringUtil::read_string(i, "input_dim") == "xxx");

        StringUtil::write_string(o, "xxx", "yyy");
        assert(o.str() == "xxx = yyy\n");
    }

    static void test_assert_next_line()
    {
        std::stringstream i;
        i << "hello\n";
        StringUtil::assert_next_line(i, "hello");
    }

    static void test_read_and_write_float_vector()
    {
        std::stringstream i, o;
        Vector<float> v({0.1, 0.3, 0}), v1;
        StringUtil::write_vector(o, v);
        assert(o.str() == "0.1 0.3 0\n");        

        i << "0.1 0.3 0.0\n";
        v1.reserve(3);
        StringUtil::read_float_vector(i, v1);
        assert(v1.equals_to({0.1, 0.3, 0}));
    }

    static void test_read_and_write_uint_vector()
    {
        std::stringstream i, o;
        Vector<uint> v({1, 3, 2}), v1;
        StringUtil::write_vector(o, v);
        assert(o.str() == "1 3 2\n");        

        i << "1 3 2\n";
        v1.reserve(3);
        StringUtil::read_uint_vector(i, v1);
        assert(v1.equals_to({1, 3, 2}));
    }
};