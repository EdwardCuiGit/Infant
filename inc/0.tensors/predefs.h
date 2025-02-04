#pragma once

// predefined keywords
#define OVERRIDE
#define USED
#define NOTUSED
#define NOTEST
#define NOGRAD
#define SEALED
#define SUGAR
#define HEAVY
#undef DEPRECATED
#undef DISABLED
#define NOTIMPLEMENTED

// predefined includes
#include <memory>
#include <iostream>
#include <limits> //std::numeric_limits
#include <cassert>
#include <tuple>
#include <map>
#include <concepts>
#include <typeinfo>
#include <sstream>
//#include <filesystem>

// predefined types
//typedef unsigned int uint;
using uint = unsigned int;
typedef void (*Func_Void)();

// predefined smart pointers
#define Ptr std::shared_ptr
#define Tuple std::tuple
#define Map std::map
#define Str std::string

// predefined consts
#define EPSILON (std::numeric_limits<double>::epsilon())
#define INF_POS (std::numeric_limits<double>::infinity())
#define INF_NEG (-std::numeric_limits<double>::infinity())

// logging
bool ___ENABLE_LOG = true;
#define ENABLE_LOGGING(yes) ___ENABLE_LOG = yes;
#define LOG_INFO(text) if (___ENABLE_LOG) std::cout << "INFO:" << text << "\n";
#define LOG_WARN(text) if (___ENABLE_LOG) std::cout << "WARN:" << text << "\n";
#define LOG_ERROR(text) if (___ENABLE_LOG) std::cout << "ERROR:" << text << "\n";
#define LOG_FATAL(text) if (___ENABLE_LOG) std::cout << "FATAL:" << text << "\n";

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
