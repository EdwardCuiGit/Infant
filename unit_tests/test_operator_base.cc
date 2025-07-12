#pragma once
#include "unit_test.h"
#include "inc/2.operators/operator_base.h"
#include "inc/2.operators/attentions.h"

class TestOperatorBase: public TestClass
{
public:
    REGISTER_TEST_CASES(test_save, test_save)

    static void test_save()
    {
        Fc fc(Fc::Config(2, 3, true, TensorInit_Types::One, TensorInit_Types::Zero));
        std::stringstream o;
        fc.save_op(o);
        std::string str = o.str();
        assert(str == "Operator Type = Fc\nOperator Id = \nstart of config\nb_type 0 1\ninput_dim 0 2\noutput_dim 0 3\nw_type 0 2\nhas_bias 2 1\nend of config\nstart of params of one operator\nparam_size = 2\nid = b\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 1\n3\n0 0 0\nend of tensor data\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n3 2\n1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\n");

        // note: we shall create a new operator with nested operators
        /*Attention att(Attention::Config(2, 3, 1, false, 1, true, 4, false, TensorInit_Types::One));
        std::stringstream o1;
        att.save_op(o1);
        std::string str1 = o1.str();
        //assert(str1 == "Operator Type = Attention\nOperator Id = \nstart of config\nbias_init_type 0 1\nfc_intermediate_factor 0 4\nhidden_dim 0 2\ninit_type 0 2\nmulti_head 0 1\nnode_len 0 3\nhas_bias 2 0\nhas_fc 2 1\nis_cross_attention 2 0\ndk 3 1\nlm 5 LayerNorm\nstart of config\naffine 2 1\nhas_lm 2 0\ntrack_running_stats 2 0\nmomentum 3 0.1\nlast_dims 4 2 \nend of config\nend of config\nstart of params of one operator\nparam_size = 4\nid = wk\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wp\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 2\n1 1 1 1\nend of tensor data\nid = wq\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wv\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nSub_Operator_Count = 5\nSub_Op_Id = fc1\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n8 2\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = fc2\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 8\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm1\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm2\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nend of params of one operator\n");
        assert(str1 == "Operator Type = Attention\nOperator Id = \nstart of config\nbias_init_type 0 1\nfc_intermediate_factor 0 4\nhidden_dim 0 2\ninit_type 0 2\nmulti_head 0 1\nnode_len 0 3\nhas_bias 2 0\nhas_fc 2 1\nis_cross_attention 2 0\ndk 3 1\nlm 5 LayerNorm\nstart of config\naffine 2 1\nhas_lm 2 0\ntrack_running_stats 2 0\nmomentum 3 0.1\nlast_dims 4 2 \nend of config\nend of config\nstart of params of one operator\nparam_size = 5\nid = right_higher_neg_inf\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n3 3\n0 -inf -inf 0 0 -inf 0 0 0\nend of tensor data\nid = wk\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wp\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 2\n1 1 1 1\nend of tensor data\nid = wq\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wv\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nSub_Operator_Count = 5\nSub_Op_Id = fc1\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n8 2\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = fc2\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 8\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm1\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm2\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nend of params of one operator\n");
        */
    }

    static void test_load()
    {
        // REGISTER_OP(Fc);
        // REGISTER_OP(Attention);

        std::stringstream i;
        i << "Operator Type = Fc\nOperator Id = \nstart of config\nb_type 0 1\ninput_dim 0 2\noutput_dim 0 3\nw_type 0 2\nhas_bias 2 1\nend of config\nstart of params of one operator\nparam_size = 2\nid = b\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 1\n3\n0 0 0\nend of tensor data\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n3 2\n1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\n";
        auto op = std::static_pointer_cast<Fc>(Operator::Load_Op(i));
        assert(op->_c.input_dim() == 2);
        assert(op->_c.has_bias() == true);
        assert(op->_w.equals_to({3, 2}, {1, 1, 1, 1, 1, 1}));

        /*std::stringstream i1;
        i1 << "Operator Type = Attention\nOperator Id = \nstart of config\nbias_init_type 0 1\nfc_intermediate_factor 0 4\nhidden_dim 0 2\ninit_type 0 2\nmulti_head 0 1\nnode_len 0 3\nhas_bias 2 0\nhas_fc 2 1\nis_cross_attention 2 0\ndk 3 1\nlm 5 LayerNorm\nstart of config\naffine 2 1\nhas_lm 2 0\ntrack_running_stats 2 0\nmomentum 3 0.1\nlast_dims 4 2 \nend of config\nend of config\nstart of params of one operator\nparam_size = 4\nid = wk\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wp\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 2\n1 1 1 1\nend of tensor data\nid = wq\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nid = wv\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 3\n1 2 2\n1 1 1 1\nend of tensor data\nSub_Operator_Count = 5\nSub_Op_Id = fc1\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n8 2\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = fc2\nstart of params of one operator\nparam_size = 1\nid = w\nis_auto_grad = 1\nis_param = 1\nis_print = 0\nstart of tensor data\nshape = 2\n2 8\n1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nend of tensor data\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm1\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nSub_Op_Id = lm2\nstart of params of one operator\nparam_size = 0\nSub_Operator_Count = 0\nend of params of one operator\nend of params of one operator\n";
        auto op1 = std::static_pointer_cast<Attention>(Operator::Load_Op(i1));
        assert(op1->_c.hidden_dim() == 2);
        assert(op1->_c.bias_init_type() == (uint)TensorInit_Types::One);
        assert(op1->_wq.equals_to({1, 2, 2}, {1, 1, 1, 1}));*/
    }
};