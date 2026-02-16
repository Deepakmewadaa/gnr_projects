#ifndef RELU_H
#define RELU_H

#include "tensor.h"

class ReLU {
public:
    Tensor input_cache;

    Tensor forward(Tensor &input);
    Tensor backward(Tensor &grad_output);
};

#endif
