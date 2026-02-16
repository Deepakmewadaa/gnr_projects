#include "optimizer.h"

SGD::SGD(std::vector<Tensor*> parameters,
         float learning_rate,
         float momentum_val)
    : params(parameters),
      lr(learning_rate),
      momentum(momentum_val) {

    velocity.resize(params.size());

    for (size_t i = 0; i < params.size(); i++) {
        velocity[i].resize(params[i]->numel(), 0.0f);
    }
}

void SGD::step() {

    for (size_t p = 0; p < params.size(); p++) {

        for (int i = 0; i < params[p]->numel(); i++) {

            velocity[p][i] =
                momentum * velocity[p][i]
                - lr * params[p]->grad[i];

            params[p]->data[i] += velocity[p][i];
        }
    }
}

void SGD::zero_grad() {

    for (auto p : params)
        for (auto &g : p->grad)
            g = 0.0f;
}
