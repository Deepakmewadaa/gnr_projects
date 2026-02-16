#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include <vector>

float cross_entropy_forward(Tensor &logits,
                            std::vector<int> &labels);

void cross_entropy_backward(Tensor &logits,
                            std::vector<int> &labels);

#endif
