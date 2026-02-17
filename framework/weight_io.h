#pragma once
#include <vector>
#include <string>
#include "tensor.h"

void save_weights(std::vector<Tensor*> params,
                  const std::string &filename);

void load_weights(std::vector<Tensor*> params,
                  const std::string &filename);
