#pragma once
#include "common_prefix.h"
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
#define CURSOR
#define NEEDS_CUDA

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
typedef void (*Func_Void)();

// predefined consts
#define EPSILON (std::numeric_limits<float>::epsilon())
#define INF_POS (std::numeric_limits<float>::infinity())
#define INF_NEG (-std::numeric_limits<float>::infinity())


// predefined smart pointers
#define Ptr std::shared_ptr
#define Tuple std::tuple
#define Map std::map
#define Str std::string


// logging
extern bool ___ENABLE_LOG;
#define ENABLE_LOGGING(yes) ___ENABLE_LOG = yes;
#define LOG_INFO(text) if (___ENABLE_LOG) std::cout << "INFO:" << text << "\n";
#define LOG_WARN(text) if (___ENABLE_LOG) std::cout << "WARN:" << text << "\n";
#define LOG_ERROR(text) if (___ENABLE_LOG) std::cout << "ERROR:" << text << "\n";
#define LOG_FATAL(text) if (___ENABLE_LOG) std::cout << "FATAL:" << text << "\n";

class Environment 
{
private:
    static bool _is_train;
    static bool _print_functor;
    static bool _enable_cuda;
public:
    static bool Is_Print_Functor()
    {
        return _print_functor;
    }

    static void Set_Print_Functor(bool yes = true)
    {
        _print_functor = yes;
    }

    static bool Is_Train()
    {
        return _is_train;
    }

    static void Set_Train(bool yes = true)
    {
        _is_train = yes;
    }

    static bool Enabled_Cuda()
    {
        return _enable_cuda;
    }

    static void Enable_Cuda(bool yes = true)
    {
        _enable_cuda = yes;
    }

    static void Init();
};
