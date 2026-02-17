#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include "framework/weight_io.h"
#include "framework/dataset.h"
#include "framework/conv2d.h"
#include "framework/relu.h"
#include "framework/maxpool.h"
#include "framework/linear.h"
#include "framework/loss.h"
#include "framework/optimizer.h"

// ACCURACY

float compute_accuracy(Tensor &logits,
                       std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    int correct = 0;

    for (int b = 0; b < batch; b++) {

        int best = 0;
        float best_val = -1e9;

        for (int c = 0; c < classes; c++) {

            float val = logits.data[b*classes + c];

            if (val > best_val) {
                best_val = val;
                best = c;
            }
        }

        if (best == labels[b])
            correct++;
    }

    return (float)correct / batch;
}

// CNN MODEL

class CNN {
public:

    Conv2D conv1;
    Conv2D conv2;
    Conv2D conv3;

    ReLU relu1, relu2, relu3;
    MaxPool2D pool1, pool2, pool3;

    Linear fc;

    CNN(int num_classes,
        int in_channels)
    :
      conv1(in_channels, 16, 3),
      conv2(16, 32, 3),
      conv3(32, 64, 3),
      pool1(2),
      pool2(2),
      pool3(2),
      fc(64*2*2, num_classes) {}

    Tensor forward(Tensor &x) {

        x = conv1.forward(x);
        x = relu1.forward(x);
        x = pool1.forward(x);

        x = conv2.forward(x);
        x = relu2.forward(x);
        x = pool2.forward(x);

        x = conv3.forward(x);
        x = relu3.forward(x);
        x = pool3.forward(x);

        int N = x.shape[0];

        Tensor flat({N, 64*2*2});

        for (int i = 0; i < x.numel(); i++)
            flat.data[i] = x.data[i];

        return fc.forward(flat);
    }

    std::vector<Tensor*> parameters() {

        return {
            &conv1.weight,&conv1.bias,
            &conv2.weight,&conv2.bias,
            &conv3.weight,&conv3.bias,
            &fc.weight,&fc.bias
        };
    }
};

// PARAM COUNT

int count_parameters(std::vector<Tensor*> params) {

    int total = 0;

    for (auto p : params)
        total += p->numel();

    return total;
}


// MACs

long long compute_macs(CNN &model,
                       int channels,
                       int input_size) {

    long long total_macs = 0;

    int H = input_size;
    int W = input_size;

    // Conv1
    {
        int K = model.conv1.kernel_size;
        int Cin = channels;
        int Cout = model.conv1.out_channels;

        int H_out = H - K + 1;
        int W_out = W - K + 1;

        total_macs +=
            (long long)H_out * W_out *
            Cout * (Cin * K * K);

        H = H_out / 2;
        W = W_out / 2;
    }

    // Conv2
    {
        int K = model.conv2.kernel_size;
        int Cin = model.conv2.in_channels;
        int Cout = model.conv2.out_channels;

        int H_out = H - K + 1;
        int W_out = W - K + 1;

        total_macs +=
            (long long)H_out * W_out *
            Cout * (Cin * K * K);

        H = H_out / 2;
        W = W_out / 2;
    }

    // Conv3
    {
        int K = model.conv3.kernel_size;
        int Cin = model.conv3.in_channels;
        int Cout = model.conv3.out_channels;

        int H_out = H - K + 1;
        int W_out = W - K + 1;

        total_macs +=
            (long long)H_out * W_out *
            Cout * (Cin * K * K);

        H = H_out / 2;
        W = W_out / 2;
    }

    total_macs +=
        (long long)model.fc.in_features *
        model.fc.out_features;

    return total_macs;
}



int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cout << "Usage: ./train <dataset_path> <num_classes>\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    int num_classes = std::stoi(argv[2]);

    ImageDataset dataset(dataset_path);

    std::cout << "Dataset size: "
              << dataset.size() << "\n";


    // Train/Test Split (80/20)
    int total_size = dataset.size();
    int train_size = (int)(0.8f * total_size);

    std::vector<int> indices(total_size);
    for (int i = 0; i < total_size; i++)
        indices[i] = i;

    std::mt19937 rng(42);
    std::shuffle(indices.begin(),
                 indices.end(),
                 rng);

    std::vector<int> train_indices(
        indices.begin(),
        indices.begin() + train_size);

    std::vector<int> test_indices(
        indices.begin() + train_size,
        indices.end());

    std::cout << "Train size: "
              << train_indices.size() << "\n";

    std::cout << "Test size: "
              << test_indices.size() << "\n";

    CNN model(num_classes,
              dataset.channels);

    std::cout << "Total Parameters: "
              << count_parameters(model.parameters())
              << "\n";

    long long macs =
        compute_macs(model,
                     dataset.channels,
                     32);

    std::cout << "MACs (per image): "
              << macs << "\n";

    std::cout << "FLOPs (per image): "
              << 2 * macs << "\n";

    SGD optimizer(model.parameters(), 0.02f, 0.9f);

    int epochs = 10;
    int batch_size = 64;

    for (int epoch = 0; epoch < epochs; epoch++) {
        

        float total_loss = 0;
        float total_acc  = 0;
        int batches = 0;


        for (int i = 0;
             i < train_size;
             i += batch_size) {

            int actual_batch =
                std::min(batch_size,
                         train_size - i);

            Tensor batch_images(
                {actual_batch,
                 dataset.channels,
                 32,32});

            std::vector<int> batch_labels;

            int img_size =
                dataset.channels * 32 * 32;

            for (int b=0; b<actual_batch; b++) {

                int idx = train_indices[i+b];

                for (int k=0;k<img_size;k++)
                    batch_images.data[b*img_size + k] =
                        dataset.images[idx].data[k];

                batch_labels.push_back(
                    dataset.labels[idx]);
            }

            optimizer.zero_grad();

            Tensor logits =
                model.forward(batch_images);

            float loss =
                cross_entropy_forward(
                    logits,
                    batch_labels);

            cross_entropy_backward(
                logits,
                batch_labels);

            Tensor grad(logits.shape);

            for (int k=0;k<logits.numel();k++)
                grad.data[k] = logits.grad[k];

            grad = model.fc.backward(grad);

            Tensor reshaped({actual_batch,64,2,2});

            for (int k=0;k<grad.numel();k++)
                reshaped.data[k] = grad.data[k];

            grad = reshaped;

            grad = model.pool3.backward(grad);
            grad = model.relu3.backward(grad);
            grad = model.conv3.backward(grad);

            grad = model.pool2.backward(grad);
            grad = model.relu2.backward(grad);
            grad = model.conv2.backward(grad);

            grad = model.pool1.backward(grad);
            grad = model.relu1.backward(grad);
            grad = model.conv1.backward(grad);

            optimizer.step();

            total_loss += loss;
            total_acc  += compute_accuracy(
                            logits,
                            batch_labels);

            batches++;
        }

        float train_acc = total_acc / batches;


        float test_acc = 0;
        int test_batches = 0;

        for (int i=0;
             i<test_indices.size();
             i+=batch_size) {

            int actual_batch =
                std::min(batch_size,
                         (int)test_indices.size() - i);

            Tensor batch_images(
                {actual_batch,
                 dataset.channels,
                 32,32});

            std::vector<int> batch_labels;

            int img_size =
                dataset.channels * 32 * 32;

            for (int b=0;b<actual_batch;b++) {

                int idx = test_indices[i+b];

                for (int k=0;k<img_size;k++)
                    batch_images.data[b*img_size + k] =
                        dataset.images[idx].data[k];

                batch_labels.push_back(
                    dataset.labels[idx]);
            }

            Tensor logits =
                model.forward(batch_images);

            test_acc += compute_accuracy(
                            logits,
                            batch_labels);

            test_batches++;
        }

        std::cout << "Epoch "
                  << epoch+1
                  << " | Loss: "
                  << total_loss/batches
                  << " | Train Acc: "
                  << train_acc*100
                  << "% | Test Acc: "
                  << (test_acc/test_batches)*100
                  << "%\n";
    }

    save_weights(model.parameters(),
                 "model.bin");

    std::cout << "Training Complete. "
              << "Weights saved to model.bin\n";

    return 0;
}
