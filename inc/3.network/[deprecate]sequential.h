#pragma once

#include "network.h"

class Sequential : public Network
{
protected:
    Array<UnOp> _ops;

    virtual TensorList build_network_core(const TensorList &x) override
    {
    }

public:
    void append(UnOp &op)
    {
        _ops.push_back(op);
    }
};
