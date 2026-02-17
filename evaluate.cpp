#include <iostream>
#include <vector>
#include "framework/dataset.h"
#include "framework/conv2d.h"
#include "framework/relu.h"
#include "framework/maxpool.h"
#include "framework/linear.h"
#include "framework/weight_io.h"

// ------------------------------------------------------------
// Accuracy
// ------------------------------------------------------------
float compute_accuracy(Tensor &logits,
                       std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    int correct = 0;

    for (int b=0; b<batch; b++) {

        int best = 0;
        float best_val = -1e9;

        for (int c=0; c<classes; c++) {

            float val =
                logits.data[b*classes + c];

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


// ------------------------------------------------------------
// SAME CNN ARCHITECTURE AS TRAINING
// ------------------------------------------------------------
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
      conv1(in_channels,16,3),
      conv2(16,32,3),
      conv3(32,64,3),
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

        for (int i=0;i<x.numel();i++)
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

int count_parameters(std::vector<Tensor*> params) {

    int total = 0;

    for (auto p : params)
        total += p->numel();

    return total;
}

long long compute_macs(CNN &model,
                       int channels,
                       int input_size) {

    long long total_macs = 0;

    int N = 1; // per single image
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

    // FC
    total_macs +=
        (long long)model.fc.in_features *
        model.fc.out_features;

    return total_macs;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main(int argc, char* argv[]) {

    if (argc < 4) {
    std::cout << "Usage:\n";
    std::cout << "./evaluate <dataset_path> <num_classes> <weights_path>\n";
    return 1;
}

    std::string dataset_path = argv[1];
    int num_classes = std::stoi(argv[2]);
    std::string weights_path = argv[3];

    ImageDataset dataset(dataset_path);

    // Optional safety check
    if (num_classes != dataset.num_classes) {
        std::cout << "Warning: num_classes mismatch. "
                << "Using dataset detected value: "
                << dataset.num_classes << "\n";
        num_classes = dataset.num_classes;
}


    CNN model(num_classes,
              dataset.channels);

    // Load trained weights
    load_weights(model.parameters(),
                 weights_path);

    // Print efficiency metrics
    int total_params =
        count_parameters(model.parameters());

    std::cout << "Total Parameters: "
            << total_params << "\n";

    long long macs =
        compute_macs(model,
                    dataset.channels,
                    32);

    std::cout << "MACs (per image): "
            << macs << "\n";

    std::cout << "FLOPs (per image): "
            << 2 * macs << "\n";

    
    int batch_size = 64;

    float total_acc = 0.0f;
    int batches = 0;

    for (int i=0;
         i<dataset.size();
         i+=batch_size) {

        Tensor batch_images;
        std::vector<int> batch_labels;

        dataset.get_batch(i,
                          batch_size,
                          batch_images,
                          batch_labels);

        Tensor logits =
            model.forward(batch_images);

        total_acc += compute_accuracy(
                        logits,
                        batch_labels);

        batches++;
    }

    std::cout << "Test Accuracy: "
              << (total_acc/batches)*100
              << "%\n";

    return 0;
}



