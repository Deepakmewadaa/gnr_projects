#pragma once
// #include "layers.h"
#include "framework/relu.h"
#include "linear.h"
#include "conv2d.h"




struct SimpleCNN {
    Conv2D conv;
    ReLU relu;
    Linear fc;

    SimpleCNN(int in_channels, int num_classes)
        : conv(in_channels, 8, 3),
          fc(8*30*30, num_classes) {}

    Tensor forward(Tensor &x) {
        auto out = conv.forward(x);
        out = relu.forward(out);

        Tensor flat({(int)out.size()});
        flat.data = out.data;

        return fc.forward(flat);
    }

    vector<Tensor*> parameters() {
        return { &conv.W, &conv.b, &fc.W, &fc.b };
    }
};
