#include "maxpool.h"

MaxPool2D::MaxPool2D(int k) : kernel(k) {}

Tensor MaxPool2D::forward(Tensor &input) {

    input_cache = input;

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int H_out = H / kernel;
    int W_out = W / kernel;

    Tensor output({N, C, H_out, W_out});

    max_indices.resize(output.numel());

    for (int n=0;n<N;n++)
    for (int c=0;c<C;c++)
    for (int h=0;h<H_out;h++)
    for (int w=0;w<W_out;w++) {

        float max_val = -1e9;
        int max_idx = -1;

        for (int kh=0;kh<kernel;kh++)
        for (int kw=0;kw<kernel;kw++) {

            int in_h = h*kernel + kh;
            int in_w = w*kernel + kw;

            int idx =
                n*C*H*W +
                c*H*W +
                in_h*W +
                in_w;

            if (input.data[idx] > max_val) {
                max_val = input.data[idx];
                max_idx = idx;
            }
        }

        int out_idx =
            n*C*H_out*W_out +
            c*H_out*W_out +
            h*W_out +
            w;

        output.data[out_idx] = max_val;
        max_indices[out_idx] = max_idx;
    }

    return output;
}

Tensor MaxPool2D::backward(Tensor &grad_output) {

    Tensor grad_input(input_cache.shape);

    for (int i=0;i<grad_output.numel();i++) {
        int idx = max_indices[i];
        grad_input.data[idx] += grad_output.data[i];
    }

    return grad_input;
}
