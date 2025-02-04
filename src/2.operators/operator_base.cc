#pragma once
#include "../../inc/2.operators/operator_base.h"

#include "../../inc/2.operators/fc.h"
#include "../../inc/2.operators/conv.h"
#include "../../inc/2.operators/norm.h"
#include "../../inc/2.operators/pooling.h"
#include "../../inc/2.operators/rnn.h"
#include "../../inc/2.operators/attentions.h"
#include "../../inc/3.network/transformer.h"

void Environment::Init()
{
    REGISTER_OP(Fc);
    REGISTER_OP(LayerNorm);
    //REGISTER_OP(BatchNorm2d);
    //REGISTER_OP(Conv2d);
    //REGISTER_OP(Pooling2d);
    //REGISTER_OP(RawRnn);
    //REGISTER_OP(Lstm);
    //REGISTER_OP(Gru);
    REGISTER_OP(Attention);
    REGISTER_OP(Transformer);
};