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
//#include <concepts>
//#include <filesystem>

// predefined types
//typedef unsigned int uint;
using uint = unsigned int;
typedef void (*Func_Void)();

// predefined smart pointers
#define Ptr std::shared_ptr

// predefined consts
#define EPSILON (std::numeric_limits<double>::epsilon())

// assert
#define ALMOST_ZERO(EXP) ((EXP) < 0.00001 && (EXP) > -0.00001)
#define assert_almost_equal(a, b) assert(ALMOST_ZERO((a) - (b)))

// logging
bool ___ENABLE_LOG = true;
#define ENABLE_LOGGING(yes) ___ENABLE_LOG = yes;
#define LOG_INFO(text) if (___ENABLE_LOG) std::cout << "INFO:" << text << "\n";
#define LOG_WARN(text) if (___ENABLE_LOG) std::cout << "WARN:" << text << "\n";
#define LOG_ERROR(text) if (___ENABLE_LOG) std::cout << "ERROR:" << text << "\n";
#define LOG_FATAL(text) if (___ENABLE_LOG) std::cout << "FATAL:" << text << "\n";