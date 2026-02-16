#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    int in_features, out_features;
    Tensor weight, bias;
    Tensor input_cache;

    Linear(int in_f, int out_f);

    Tensor forward(Tensor &input);
    Tensor backward(Tensor &grad_output);
};

#endif
