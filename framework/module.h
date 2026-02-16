#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"
#include <vector>
#include <memory>

class Module {
public:
    virtual Tensor forward(Tensor &input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
};

#endif
