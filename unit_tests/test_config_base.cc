#pragma once
#include "unit_test.h"
#include "../inc/2.operators/config_base.h"
#include "../inc/2.operators/norm.h"
#include "../inc/2.operators/operator_base.h"

class TestConfigBase: public TestClass
{
public:
    REGISTER_TEST_CASES(test)

    static void test()
    {
        // REGISTER_OP(LayerNorm);

        struct Config : ConfigBase
        {
            DEFINE_FIELD(uint, input_dim, 100);
            DEFINE_FIELD(bool, has_bias, true);
            DEFINE_FIELD(int, a, -1);
            DEFINE_FIELD(double, b, 0.2);
            Vector<uint> &last_dims() { return access_uint_vector("last_dims"); } 
            const Vector<uint> last_dims() const { return access_uint_vector("last_dims"); } 
            DEFINE_SUB_CONFIG(LayerNorm, lm);

            Config() : ConfigBase("TestConfig")
            {}
        };

        Config c;
        assert(c.input_dim() == 100);
        c.input_dim() = 200;
        assert(c.input_dim() == 200);
        assert(c.has_bias() == true);
        c.has_bias() = false;
        assert(c.has_bias() == false);
        assert(c.a() == -1);
        c.a() = -2;
        assert(c.a() == -2);
        assert(c.b() == 0.2);
        c.b() = 0.3;
        assert(c.b() == 0.3);
        assert(c.last_dims().equals_to({}));
        c.last_dims().append({0, 2, 1});
        assert(c.last_dims().equals_to({0, 2, 1}));
        assert(c.lm().affine() == true);
        assert(c.lm().momentum() == 0.1);

        std::stringstream o, o1;
        c.save(o);
        auto str = o.str();
        auto expected_str = "start of config\n"
        "input_dim 0 200\n"
        "a 1 -2\n"
        "has_bias 2 0\n"
        "b 3 0.3\n"
        "last_dims 4 0 2 1 \n"
        "lm 5 LayerNorm\n"
        "start of config\n"
        "affine 2 1\n"
        "has_lm 2 0\n"
        "track_running_stats 2 0\n"
        "momentum 3 0.1\n"
        "last_dims 4 \n"
        "end of config\n"
        "end of config\n";

        assert(o.str() == expected_str);

        std::stringstream i;
        
        i << expected_str;
        Config c1;
        c1.load(i);
        c1.save(o1);
        auto o1str = o1.str();
        assert(o1.str() == expected_str);
    }
};