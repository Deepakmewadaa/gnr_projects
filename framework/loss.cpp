#include "loss.h"
#include <cmath>
#include <algorithm>

float cross_entropy_forward(Tensor &logits,
                            std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    float loss = 0.0f;

    for (int b = 0; b < batch; b++) {

 
        float max_val = -1e9f;
        for (int c = 0; c < classes; c++) {
            max_val = std::max(max_val,
                logits.data[b*classes + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; c++) {
            sum_exp += std::exp(
                logits.data[b*classes + c] - max_val);
        }

        float log_prob =
            logits.data[b*classes + labels[b]]
            - max_val
            - std::log(sum_exp);

        loss -= log_prob;
    }

    return loss / batch;
}


void cross_entropy_backward(Tensor &logits,
                            std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    for (int b = 0; b < batch; b++) {

        float max_val = -1e9f;
        for (int c = 0; c < classes; c++) {
            max_val = std::max(max_val,
                logits.data[b*classes + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; c++) {
            sum_exp += std::exp(
                logits.data[b*classes + c] - max_val);
        }

        for (int c = 0; c < classes; c++) {

            float softmax =
                std::exp(logits.data[b*classes + c] - max_val)
                / sum_exp;

            float grad = softmax;

            if (c == labels[b])
                grad -= 1.0f;

            logits.grad[b*classes + c] =
                grad / batch;
        }
    }
}
