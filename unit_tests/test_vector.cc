#pragma once
#include "unit_test.h"
#include "../inc/0.tensors/vector.h"

// for any new test case, pls add func name to main()
class TestVector : public TestClass
{
public:
    REGISTER_TEST_CASES(test_init, test_ordinal, test_rand, test_add, test_multi, test_dot, test_mse, test_ce, test_bce, test_re, test_cosine_distance, test_euclidean, \
    test_match_bottom, test_linear, test_softmax, test_sum, test_product, test_avg, test_max, test_min, test_norm_ln, test_norm_l1, \
    test_entropy, test_binary_entropy, test_all_pos, test_all_in01)

    static void test_init()
    {
        Vector<double> v;
        v.reserve(10);
        v.init(TensorInit_Types::One);
        assert(v.sum() == 10);

        Vector<double> v1(10, TensorInit_Types::Zero);
        assert(v1.sum() == 0);

        v.init(TensorInit_Types::Gaussian);
        v.print();

        v.init(TensorInit_Types::Rand);
        v.print();

        v1.init(TensorInit_Types::One);
        Vector<double> v2(v1);
        v2[1] = 2;
        assert(v2.sum() == 11);
    }

    static void test_ordinal()
    {
        Vector<double> v;
        v.reserve(5);
        v.set_ordinal();
        assert(v.equals_to({0, 1, 2, 3, 4}));
    }

    static void test_rand()
    {
        Vector<double> v;
        v.reserve(10);
        v.set_dist_student_t(5);
        v.print();

        v.set_dist_chi_squared(6);
        v.print();

        v.set_dist_cauchy(1, 2);
        v.set_dist_fisher_f(3, 4);
        v.set_dist_lognormal(5, 6);
    }

    static void test_add()
    {
        Vector<double> v1{1, 2, 3, 5, 2};
        Vector<double> v2{3, 3, 2, 1, 3, 4};
        Vector<double> out1(5), out2(3);

        v1.add(v2, out1, 0, 0, 0, 5);
        assert(out1.equals_to(Vector<double>{4, 5, 5, 6, 5}));

        v1.add(v2, out2, 2, 1, 0, 3, false, 2, 3, 1);
        assert(out2.equals_to(Vector<double>{16, 17, 8}));

        v1.add(v2, out1, 2, 1, 0, 3, true, 2, 3, 1);
        assert(out1.equals_to(Vector<double>{16+4, 17+5, 8+5, 6, 5}));

        v1.add_(v2, 2, 2, 3);
        assert(v1.equals_to(Vector<double>{1, 2, 3+2, 5+1, 2+3}));

        // test add_grad
        Vector<double> v3{1, 2, 3, 5};
        Vector<double> out3(4);
        v3.add(Vector<double>{2, 3, 4, 5}, out3, 0, 0, 0, -1, false, 2);
        Vector<double> out_grad{1, 1, 2, 2}, v3_grad(4);
        //v3.add_grad(out_grad, 2, v3_grad);
        //assert(v3_grad.equals_to(Vector<double>{2, 2, 4, 4}));
    }

    static void test_multi()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, 3, 2, 1};
        v1.mul_(v2);
        assert(v1.equals_to(Vector<double>({4, 6, 6, 5})));

        Vector<double> out(4, TensorInit_Types::One);
        v1.mul(v2, out, 0, 1, 1, 3);
        assert(out.equals_to(Vector<double>({1, 12, 12, 6})));

        // test add_grad
        Vector<double> v3{1, 2, 3, 5};
        Vector<double> v4{2, 3, 4, 5};
        Vector<double> out3(4);
        v3.mul(v4, out3, 0, 0, 0, -1, false, 2);
        Vector<double> out_grad{1, 1, 2, 2}, v3_grad(4);
        // v3.mul_grad(out_grad, v4, 2, v3_grad);
        // assert(v3_grad.equals_to(Vector<double>{4, 6, 16, 20}));
    }

    static void test_dot()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, 3, 2, 1};
        double res = v1.dot(v2);
        assert(res == 21);
    }

    static void test_mse()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, 3, 2, 1};
        double res = v1.mse(v2);
        assert(ALMOST_ZERO(res - 27 / 4.0));
    }

    static void test_ce()
    {
        Vector<double> v1{0.1, 0.2, 0.4, 0.5, 0.6};
        Vector<double> v2{0.2, 0.2, 0.3, 0.4, 0.1};

        assert_almost_equal(v1.ce(v2), 4.04548557050879);
    }

    static void test_bce()
    {
        Vector<double> v1{1, 0, 0, 1, 0};
        Vector<double> v2{0.1, 0.2, 0.4, 0.5, 0.6};
        auto res = v1.bce(v2);
        assert_almost_equal(res, 1.3405499757656585);
    }

    static void test_re()
    {
        Vector<double> v1{0.1, 0.2, 0.4, 0.5, 0.6};
        Vector<double> v2{0.2, 0.2, 0.3, 0.4, 0.1};
        auto res = v1.re(v2);
        assert_almost_equal(res, 1.7779565475879124);
    }

    static void test_cosine_distance()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, 3, 2, 1};
        double res = v1.cosine_distance(v2);
        ALMOST_ZERO(res - 21 / std::sqrt(39.0) / std::sqrt(30.0));
    }

    static void test_euclidean()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, 3, 2, 1};
        double res = v1.euclidean(v2);
        ALMOST_ZERO(res - std::sqrt(27));
    }

    static void test_match_bottom()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{2, 3, 5};
        Vector<double> v3{4, 2, 3, 5};

        assert(v1.match_bottom(v2));
        assert(v1.match_bottom(v1));
        assert(!v1.match_bottom(v3));
        assert(v1.match_bottom(v3, 1));
        assert(v1.match_bottom(v3, 2));
        assert(v1.match_bottom(v3, 3));
    }

    static void test_linear()
    {
        Vector<double> v1{1, 2, 3, 5};
        assert(v1.linear_(0, -1, 2, 1).equals_to(Vector<double>{3, 5, 7, 11}));
    }

    static void test_softmax()
    {
        Vector<double> v1{1, 2, 3};
        double denominator = std::exp(1) + std::exp(2) + std::exp(3);
        Vector<double> res{std::exp(1) / denominator, std::exp(2) / denominator, std::exp(3) / denominator};
        v1.softmax_();
        assert(v1.equals_to(res));
    }

    static void test_sum()
    {
        Vector<double> v1{1, 2, 3, 5};
        assert(v1.sum(1, 3) == 10);
        assert(v1.sum(2, 2) == 8);
        assert(v1.sum() == 11);
    }

    static void test_product()
    {
        Vector<double> v1{1, 2, 3, 5};
        assert(v1.product() == 30);
    }

    static void test_avg()
    {
        Vector<double> v1{1, 2, 3, 5};
        auto res = v1.avg(1, 3);
        assert(res == 10 / 3.0);
    }

    static void test_max()
    {
        Vector<double> v1{1, 2, 3, 5};
        assert(v1.max(0, v1.size()) == 5);
    }

    static void test_min()
    {
        Vector<double> v1{1, 2, 3, 5};
        assert(v1.min(1, 2) == 2);
    }

    static void test_norm_ln()
    {
        Vector<double> v1{2, 3, 4, 3};
        auto res = v1.norm_ln(2, 0, v1.size());
        assert(ALMOST_ZERO(res - 6.164414002968976));
    }

    static void test_norm_l1()
    {
        Vector<double> v1{2, -3, -4, 3};
        assert(v1.norm_l1(0, v1.size()) == 12);
    }

    static void test_entropy()
    {
        Vector<double> v1{0.1, 0.2, 0.4, 0.5, 0.6};
        auto res = v1.entropy(0, v1.size());
        assert_almost_equal(res, 2.2675290229208773);
    }

    static void test_binary_entropy()
    {
        assert_almost_equal(Vector<double>::binary_entropy(0.2), 0.721928);
    }

    static void test_all_pos()
    {
        Vector<double> v1{1, 2, 3, 5};
        Vector<double> v2{4, -3, 2, 1};

        assert(v1.all_pos(0, 4));
        assert(!v2.all_pos(1, 3));
    }

    static void test_all_in01()
    {
        Vector<double> v1{0.1, 0.5, 1.0, 0};
        Vector<double> v2{1.2, 0.5, 0.5, 1.0};

        assert(v1.all_in01(false, 0, 4));
        assert(!v2.all_in01(false, 0, 4));
        assert(v2.all_in01(false, 1, 3));
        assert(!v2.all_in01(true, 1, 3));
    }
};