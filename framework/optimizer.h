#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>

class SGD {
public:
    std::vector<Tensor*> params;
    std::vector<std::vector<float>> velocity;

    float lr;
    float momentum;

    SGD(std::vector<Tensor*> parameters,
        float learning_rate,
        float momentum_val = 0.9f);

    void step();
    void zero_grad();
};

#endif
