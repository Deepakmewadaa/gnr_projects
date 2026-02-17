#include "relu.h"
#include <algorithm>
// RELU
Tensor ReLU::forward(Tensor &input) {
    input_cache = input;

    Tensor output(input.shape);

    for (int i=0;i<input.numel();i++)
        output.data[i] =
            std::max(0.0f, input.data[i]);

    return output;
}

Tensor ReLU::backward(Tensor &grad_output) {
    Tensor grad_input(input_cache.shape);

    for (int i=0;i<input_cache.numel();i++)
        if (input_cache.data[i] > 0)
            grad_input.data[i] =
                grad_output.data[i];

    return grad_input;
}
