#pragma once
#include "unit_test.h"
#include "inc/1.functors/functor_graph.h"
#include "inc/1.functors/tensor_node.h"
#include "inc/2.operators/conv.h"
#include "inc/2.operators/fc.h"
#include "inc/2.operators/pooling.h"

class TestFunctorGraph : public TestClass
{
public:
    REGISTER_TEST_CASES(test)

    static void test()
    {
        FunctorGraph::singleton().start_auto_grad();

        Tensor x({2, 2, 3, 3}, TensorInit_Types::Ordinal);
         /*
        0, 1, 2,    9, 10, 11,
        3, 4, 5,    12,13, 14,
        6, 7, 8,    15,16, 17,

        18, 19, 20,  27,28, 29,
        21, 22, 23,  30,31, 32,
        24, 25, 26,  33,34, 35
        */
       /*x_grad:
       0.75, 1.5, 0.75,
       1.5, 3, 1.5,
       0.75, 1.5, 0.75,
       */

        /*conv k_grad: 
        0, 0, 0, 0, 0, 0, 0, 0
        0.25*(8+80, 12+84, 20+92, 24+96,    44+116, 48+120, 56+128, 60+132)
        0.5*(8+80, 12+84, 20+92, 24+96,    44+116, 48+120, 56+128, 60+132)
        */

        Tensor y1 = Conv2d({2, 3, 2, 2, 1, 1, 0, 0, 1, false, TensorInit_Types::One}).forward(x);
        /*
        52, 60,   52, 60,    52,60,
        76, 84,   76, 84,    76,84,

        52+144, 60+144,   52+144, 60+144,    52+144,60+144,
        76+144, 84+144,   76+144, 84+144,    76+144,84+144,
        */
        /*y1_grad:
        0, 0,     0.25, 0.25,    0.5, 0.5,
        0, 0,     0.25, 0.25,    0.5, 0.5

        0, 0,     0.25, 0.25,    0.5, 0.5,
        0, 0,     0.25, 0.25,    0.5, 0.5
        */

        Tensor y2 = Pooling2d({Pooling_Types::Avg, 2, 2}).forward(y1);
        /*
        68, 68, 68,
        212, 212, 212
        */
        /*y2_grad: 0, 1, 2,   0, 1, 2*/
        Tensor y3 = Fc({3, 1, true, TensorInit_Types::Ordinal, TensorInit_Types::One}).forward(y2);
        /*
        205,
        637,
        */
        /*y3_grad: 1, 1*/
        /*fc k_grad: 68+212, 68+212, 68+212*/
        /*fc b_grad: 2, 2, 2*/

        // test graph creation
        FunctorGraph &g = FunctorGraph::singleton();

        assert(g._tensors.size() == 15);
        assert(g._tensors[0] == x); 
        assert(g._tensors[6] == y1); // x, col, _k, y, y, y, y(y1)
        assert(g._tensors[10] == y2); // col, y1, y2, y(y2)
        assert(g._tensors[14] == y3); // _w, y, _b, y, y(y3)

        assert(g._params.size() == 3);
        assert(g._params[0].dim().equals_to({1, 3, 2, 2, 2}));
        assert(g._params[1].dim().equals_to({1, 3}));
        assert(g._params[2].dim().equals_to({1}));

        assert(g._functors.size() == 11);
        assert(g._functors[0]->func->type() == "Im2Col");
        assert(g._functors[0]->inputs.size() == 1);
        assert(g._functors[1]->func->type() == "Dot");
        assert(g._functors[1]->inputs.size() == 2);
        assert(g._functors[2]->func->type() == "Swap");
        assert(g._functors[2]->outputs.size() == 1);
        assert(g._functors[3]->func->type() == "MoveForward");
        assert(g._functors[4]->func->type() == "MergeDim");
        assert(g._functors[5]->func->type() == "Im2Col");
        assert(g._functors[6]->func->type() == "Avg");
        assert(g._functors[7]->func->type() == "MoveForward");
        assert(g._functors[8]->func->type() == "Squeeze");
        assert(g._functors[9]->func->type() == "Dot");
        assert(g._functors[10]->func->type() == "Add");

        assert(y3.data().vector().equals_to({205, 637}));
        assert(y1.dim().equals_to({2, 3, 2, 2}));
        assert(y2.dim().equals_to({2, 3}));
        assert(y3.dim().equals_to({2, 1}));
        // test second forward with same x
        g.zero_features();
        assert(y3.data().vector().equals_to({}));
        assert(y1.dim().equals_to({}));
        assert(y2.dim().equals_to({}));
        assert(y3.dim().equals_to({}));

        x.reset({2, 2, 3, 3}, TensorInit_Types::Ordinal);
        g.set_inputs({x});
        g.forward({x});
        assert(y3.data().vector().equals_to({205, 637}));
        assert(y1.dim().equals_to({2, 3, 2, 2}));
        assert(y2.dim().equals_to({2, 3}));
        assert(y3.dim().equals_to({2, 1}));

        // test backward()
        y3.grad().reset({2}, TensorInit_Types::One);
        g.set_outputs({y3});
        g.backward({y3});
        assert(x.grad().vector().equals_to({0.75, 1.5, 0.75, 1.5, 3, 1.5, 0.75, 1.5, 0.75,
        0.75, 1.5, 0.75, 1.5, 3, 1.5, 0.75, 1.5, 0.75,
        0.75, 1.5, 0.75, 1.5, 3, 1.5, 0.75, 1.5, 0.75,
        0.75, 1.5, 0.75, 1.5, 3, 1.5, 0.75, 1.5, 0.75,
        }));
        assert(x.grad().dim().equals_to({2, 2, 3, 3}));
        assert(y1.grad().dim().equals_to({2, 3, 2, 2}));
        assert(y2.grad().dim().equals_to({2, 3}));
        assert(y3.grad().dim().equals_to({2}));

        assert(g._params[0].grad().dim().equals_to({1, 3, 2, 2, 2}));
        assert(g._params[0].grad().vector().equals_to(
        {0, 0, 0, 0, 0, 0, 0, 0,
        0.25*(8+80), 0.25*(12+84), 0.25*(20+92), 0.25*(24+96),
        0.25*(44+116), 0.25*(48+120), 0.25*(56+128), 0.25*(60+132),
        0.5*(8+80), 0.5*(12+84), 0.5*(20+92), 0.5*(24+96),    
        0.5*(44+116), 0.5*(48+120), 0.5*(56+128), 0.5*(60+132)}
        ));

        /*fc k_grad: 68+212, 68+212, 68+212*/
        /*fc b_grad: 2, 2, 2*/
        assert(g._params[1].grad().dim().equals_to({1, 3}));
        assert(g._params[1].grad().vector().equals_to({68+212, 68+212, 68+212}));
        assert(g._params[2].grad().dim().equals_to({1}));
        assert(g._params[2].grad().vector().equals_to({2}));
    }
};