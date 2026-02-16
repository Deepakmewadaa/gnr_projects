#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"

class Conv2D {
public:
    int in_channels, out_channels, kernel_size;
    Tensor weight, bias;
    Tensor input_cache;

    Conv2D(int in_ch, int out_ch, int k);

    Tensor forward(Tensor &input);
    Tensor backward(Tensor &grad_output);
};

#endif
