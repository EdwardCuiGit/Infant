#ifdef __UNDEFINED__
#pragma once

#include "2.operators/operator_base.h"
#include <iostream>

struct InferenceConfig
{
public:
    uint mini_batch;
};

class Model
{
public:
    virtual void load_model() = 0;
    virtual bool run(DataLoadersManager &dlm, std::ostream &out, const InferenceConfig &config)
    {
        // similar to train, but just needs one eporch, and no need to calculate loss, backward, and param_upate()
        TensorPtr start_x;
        TensorPtr end_y;
        while (true)
        {
            uint samples = dlm[0]->read_next(start_x, config.mini_batch);
            if (samples > 0)
            {
                _network.forward(start_x, end_y); // needs to disable last loss layer
                _network.zero_feature();
                end_y->save(out);
            }
            else
            {
                break;
            }
        }

        return true;
    }
};
#endif