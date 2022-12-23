#pragma once

#include "2.operators/operator_base.h"

/*
ParamNorm is used to normalize parameters in one operator
x: vector, change the x to have L2_norm == g

https://arxiv.org/pdf/1602.07868.pdf
could use together with mean only batch-norm
*/