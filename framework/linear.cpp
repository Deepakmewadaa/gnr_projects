#include "linear.h"
#include <cstdlib>

Linear::Linear(int in_f, int out_f)
    : in_features(in_f),
      out_features(out_f),
      weight({out_f, in_f}),
      bias({out_f}) {

    float fan_in = in_features;
    float scale = sqrt(2.0f / fan_in);

    for (auto &w : weight.data)
        w = ((float)rand()/RAND_MAX * 2 - 1) * scale;

}

Tensor Linear::forward(Tensor &input) {

    input_cache = input;
    int batch = input.shape[0];

    Tensor output({batch, out_features});

    for (int b=0;b<batch;b++)
    for (int o=0;o<out_features;o++) {

        float sum = bias.data[o];

        for (int i=0;i<in_features;i++)
            sum += input.data[b*in_features+i] *
                   weight.data[o*in_features+i];

        output.data[b*out_features+o] = sum;
    }

    return output;
}

Tensor Linear::backward(Tensor &grad_output) {

    int batch = input_cache.shape[0];
    Tensor grad_input(input_cache.shape);

    for (auto &g : weight.grad) g = 0.0f;
    for (auto &g : bias.grad) g = 0.0f;

    for (int b=0;b<batch;b++)
    for (int o=0;o<out_features;o++) {

        float grad_out =
            grad_output.data[b*out_features+o];

        bias.grad[o] += grad_out;

        for (int i=0;i<in_features;i++) {

            weight.grad[o*in_features+i] +=
                input_cache.data[b*in_features+i] *
                grad_out;

            grad_input.data[b*in_features+i] +=
                weight.data[o*in_features+i] *
                grad_out;
        }
    }

    return grad_input;
}
