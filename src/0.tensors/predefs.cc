#include "inc/0.tensors/predefs.h"

bool ___ENABLE_LOG;
bool Environment::_is_train = false;
bool Environment::_print_functor = true;
#ifdef ENABLE_CUDA
bool Environment::_enable_cuda = true;
#else
bool Environment::_enable_cuda = false;
#endif