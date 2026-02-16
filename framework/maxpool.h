#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "tensor.h"
#include <vector>

class MaxPool2D {
public:
    int kernel;
    Tensor input_cache;
    std::vector<int> max_indices;

    MaxPool2D(int k);

    Tensor forward(Tensor &input);
    Tensor backward(Tensor &grad_output);
};

#endif
