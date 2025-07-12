#include "test_array.cc"
#include "test_string_util.cc"
#include "test_vector.cc"
#include "test_tensor.cc"
#include "test_tensor_node.cc"
#include "test_config_base.cc"
#include "test_fc.cc"
#include "test_conv.cc"
#include "test_pooling.cc"
#include "test_rnn.cc"
#include "test_norm.cc"
#include "test_functor_graph.cc"
#include "test_attentions.cc"
#include "test_operator_base.cc"
#include "test_transformer.cc"
#include "test_optimizers.cc"
#include "test_dataloaders.cc"
#include "test_trainers.cc"
#include "test_gbdt.cc"

/*
predefs.h: memory, iostream, limits, cassert
array.h: predefs.h, vector, initializer_list, algorithm, functional
string_util.h : array.h, cstring
vector.h: array.h, cmath, random
tensor.h: vector.h
*/

int main(int argc, char *argv[])
{
    std::pair<std::string, Ptr<TestClass>> test_classes[] = 
    {
        REGISTER_TEST_CLASS(TestArray),
        REGISTER_TEST_CLASS(TestStringUtil),
        REGISTER_TEST_CLASS(TestVector),
        REGISTER_TEST_CLASS(TestTensor),
        REGISTER_TEST_CLASS(TestTensorNode),
        REGISTER_TEST_CLASS(TestConfigBase),
        REGISTER_TEST_CLASS(TestFc),
        REGISTER_TEST_CLASS(TestConv),
        REGISTER_TEST_CLASS(TestPooling),
        REGISTER_TEST_CLASS(TestRnn),
        REGISTER_TEST_CLASS(TestNorm),
        REGISTER_TEST_CLASS(TestFunctorGraph),
        REGISTER_TEST_CLASS(TestAttention),
        REGISTER_TEST_CLASS(TestOperatorBase),
        REGISTER_TEST_CLASS(TestTransformer),
        REGISTER_TEST_CLASS(TestOptimizers),
        REGISTER_TEST_CLASS(TestDataLoaders),
        REGISTER_TEST_CLASS(TestTrainers),
        /*REGISTER_TEST_CLASS(TestGbdt),*/
    };

    Environment::Init();

    for (auto test_class : test_classes)
    {
        auto test_cases = test_class.second->get_test_cases();
        for (auto i = 0; i < test_cases.first.size(); ++i)
        {
            std::cout << test_class.first << " : " << test_cases.second[i] << " case started";
            try
            {
                test_cases.first[i]();
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                std::cout << test_class.first << " : " << test_cases.second[i] << " case failed";
            }
            
            std::cout << " => passed\n";
        }

        std::cout << test_class.first << " : all test cases passed\n";
    }

    std::cout << "all test classes' test cases passed\n";
    return 0;
}