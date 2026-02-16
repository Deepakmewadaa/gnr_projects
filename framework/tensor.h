#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;

    Tensor();
    Tensor(std::vector<int> shape);
    

    int numel() const;
    void zero_grad();

};

#endif
