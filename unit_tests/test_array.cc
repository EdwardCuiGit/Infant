#pragma once
#include "../inc/0.tensors/array.h"
#include "unit_test.h"
#include <iostream>
#include <sstream>

class TestArray : public TestClass
{
public:
    REGISTER_TEST_CASES(test_set_funcs, test_get_funcs, test_update_funcs)

    static void test_set_funcs()
    {
        Array<double> a1;
        assert(a1.size() == 0);
        a1.push_back(4);
        assert(a1.size() == 1);
        a1.clear();
        assert(a1.size() == 0);
        a1.push_back(4);
        a1.push_back(5);
        assert(a1.size() == 2);

        Array<double> a2{1, 2, 3};
        assert(a2[0] == 1);
        assert(a2[2] == 3);
        assert(a2.size() == 3);
        a2.append(a1);
        assert(a2[4] == 5);

        Array<double> a3(3);
        assert(a3.size() == 3);
        a3[2] = 1;
        a3.copy(a2, 1, 2);
        assert(a3[1] == 3);
        assert(a3.size() == 2);
        a3.copy(a2);
        assert(a3.size() == 5);
        a3.copy(a2, 4);
        assert(a3[0] == 5);
        a3.reserve(3);
        assert(a3.size() == 3);
        a3.reserve(6);
        assert(a3[5] == 0);
        a3[4] = 2;
        a3.set(5, a2, 2, 1);
        assert(a3[5] == 3);

        a3.set(1, a2, 2);
        assert(a3[2] == 4);

        a3[2] = 1.5;
        a3.set_each(1.6, 2, 2);
        assert(a3[3] == 1.6);

        Array<double> a4(3);
        a4.set(0, {0, 1, 2});
        a4.insert(1, {2, 3});
        assert(a4.equals_to({0, 2, 3, 1, 2}));

        Array<double> a5{1, 2};
        a5.append(a4, 1, 3);
        assert(a5.equals_to({1, 2, 2, 3, 1}));

        a5.erase(3);
        assert(a5.equals_to({1, 2, 2}));
        a5.erase(1, 1);
        assert(a5.equals_to({1, 2}));
    }

    static void test_get_funcs()
    {
        Array<int> a1{1, 2, 3, 4, 5};
        Array<int> a2{2, 3, 4};

        uint sum = 0;
        for (auto e : a1)
        {
            sum += e;
        }

        assert(sum == 15);
        assert(a1[2] == 3);
        assert(a1.front() == 1);
        assert(a1.back() == 5);

        auto res = a1.subset(1, 3);

        assert(res.equals_to(a2));
        assert(res == a2);
        assert(res != a1);

        assert(a1.find(3) == 2);
        assert(a1.find(6) == 5);
        assert(a1.find([](const int &e)->bool{return e == 4;}) == 3);
        assert(a1.find([](const int &e)->bool{return e != 4;}) == 0);
        assert(a1.find([](const int &e)->bool{return e == 6;}) == 5);

        assert(a1.contains(2));
        assert(!a1.contains(6));

        assert(a1.size() == 5);

        std::ostringstream os;
        a1.print(os);
        assert(os.str() == "1 2 3 4 5 \n");
    }

    static void test_update_funcs()
    {
        Array<int> a1{1, 2, 3, 4, 5};
        a1.swap(2, 4);
        assert(a1[2] == 5);
        assert(a1[4] == 3);

        a1.move_forward(3, 2, 1);
        assert(a1.equals_to(Array<int>{1, 4, 3, 2, 5}));

        int sum = 0;
        a1.loop([&sum](uint e)->void{sum += e;});
        assert(sum == 15);
    }
};
