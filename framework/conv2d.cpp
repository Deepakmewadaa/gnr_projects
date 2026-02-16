#include "conv2d.h"
#include <cstdlib>

Conv2D::Conv2D(int in_ch, int out_ch, int k)
    : in_channels(in_ch),
      out_channels(out_ch),
      kernel_size(k),
      weight({out_ch, in_ch, k, k}),
      bias({out_ch}) {

    float fan_in = in_channels * kernel_size * kernel_size;
    float scale = sqrt(2.0f / fan_in);

    for (auto &w : weight.data)
        w = ((float)rand()/RAND_MAX * 2 - 1) * scale;

}

Tensor Conv2D::forward(Tensor &input) {

    input_cache = input;

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int K = kernel_size;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    Tensor output({N, out_channels, H_out, W_out});

    for (int n=0;n<N;n++)
    for (int oc=0;oc<out_channels;oc++)
    for (int h=0;h<H_out;h++)
    for (int w=0;w<W_out;w++) {

        float sum = bias.data[oc];

        for (int ic=0;ic<in_channels;ic++)
        for (int kh=0;kh<K;kh++)
        for (int kw=0;kw<K;kw++) {

            int in_idx =
                n*C*H*W + ic*H*W +
                (h+kh)*W + (w+kw);

            int w_idx =
                oc*in_channels*K*K +
                ic*K*K +
                kh*K + kw;

            sum += input.data[in_idx] *
                   weight.data[w_idx];
        }

        int out_idx =
            n*out_channels*H_out*W_out +
            oc*H_out*W_out +
            h*W_out + w;

        output.data[out_idx] = sum;
    }

    return output;
}

Tensor Conv2D::backward(Tensor &grad_output) {

    int N = input_cache.shape[0];
    int C = input_cache.shape[1];
    int H = input_cache.shape[2];
    int W = input_cache.shape[3];

    int K = kernel_size;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    Tensor grad_input(input_cache.shape);

    for (auto &g : weight.grad) g = 0.0f;
    for (auto &g : bias.grad) g = 0.0f;

    for (int n=0;n<N;n++)
    for (int oc=0;oc<out_channels;oc++)
    for (int h=0;h<H_out;h++)
    for (int w=0;w<W_out;w++) {

        int out_idx =
            n*out_channels*H_out*W_out +
            oc*H_out*W_out +
            h*W_out + w;

        float grad_out = grad_output.data[out_idx];
        bias.grad[oc] += grad_out;

        for (int ic=0;ic<in_channels;ic++)
        for (int kh=0;kh<K;kh++)
        for (int kw=0;kw<K;kw++) {

            int in_idx =
                n*C*H*W + ic*H*W +
                (h+kh)*W + (w+kw);

            int w_idx =
                oc*in_channels*K*K +
                ic*K*K +
                kh*K + kw;

            weight.grad[w_idx] +=
                input_cache.data[in_idx] * grad_out;

            grad_input.data[in_idx] +=
                weight.data[w_idx] * grad_out;
        }
    }

    return grad_input;
}
