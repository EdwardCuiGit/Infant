#pragma once

#include <float.h>
#include <stdio.h>

#undef ENABLE_CUDA

using uint = unsigned int;

enum class TensorInit_Types : uint
{
    None,
    Zero,
    One,
    Half,
    Ordinal,
    Rand,
    Gaussian,
    LEFT_LOWER_ONE,
    RIGHT_HIGHER_NEG_INF,
};

enum class Activation_Types : uint
{
    None,
    Linear,
    Relu,
    Tanh,
    Sigmoid
};

enum class Pooling_Types: uint
{
    Avg,
    Max,
    Min
};

enum class NumTypes : uint
{
    Uint = 0,
    Int,
    Bool,
    Double,
    UintVector,
    SubConfig,
    Total
};

enum class CompareTypes : uint
{
    Equal = 0,
    Not_Equal,
    Greater_Than,
    Greater_Equal_Than,
    Less_Than,
    Less_Equal_Then,
    Total
};

enum class NormTypes : uint
{
    BatchNorm = 0,
    LayerNorm,
    RMSNorm,
    Total
};
