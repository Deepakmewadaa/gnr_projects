#include "tensor.h"
#include <algorithm>

Tensor::Tensor() {}

Tensor::Tensor(std::vector<int> shape)
    : shape(shape) {

    int total = 1;
    for (int s : shape)
        total *= s;

    data.resize(total, 0.0f);
    grad.resize(total, 0.0f);
}

int Tensor::numel() const {

    int total = 1;
    for (int s : shape)
        total *= s;

    return total;
}

void Tensor::zero_grad() {

    std::fill(grad.begin(),
              grad.end(),
              0.0f);
}
