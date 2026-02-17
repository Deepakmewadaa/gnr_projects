#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include "framework/dataset.h"
#include "framework/conv2d.h"
#include "framework/relu.h"
#include "framework/maxpool.h"
#include "framework/linear.h"
#include "framework/loss.h"
#include "framework/optimizer.h"
#include "framework/weight_io.h"

namespace py = pybind11;

// ---------------- CNN ----------------

class CNN {
public:
    Conv2D conv1;
    Conv2D conv2;
    Conv2D conv3;

    ReLU relu1, relu2, relu3;
    MaxPool2D pool1, pool2, pool3;

    Linear fc;

    CNN(int num_classes, int in_channels)
        : conv1(in_channels,16,3),
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

        for(int i=0;i<x.numel();i++)
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

// ---------------- Accuracy ----------------

float compute_accuracy(Tensor &logits,
                       std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    int correct = 0;

    for (int b=0; b<batch; b++) {
        int best = 0;
        float best_val = -1e9;

        for (int c=0; c<classes; c++) {
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

// ---------------- Train ----------------

float train_model(const std::string &dataset_path,
                  int num_classes,
                  int epochs,
                  int batch_size,
                  const std::string &weights_path) {

    // ---------------- Dataset Loading ----------------

    ImageDataset dataset(dataset_path);

    // ---------------- Train/Test Split ----------------
    int total_size = dataset.size();
    int train_size = (int)(0.8f * total_size);

    std::vector<int> indices(total_size);
    for (int i = 0; i < total_size; i++)
        indices[i] = i;

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<int> train_indices(
        indices.begin(),
        indices.begin() + train_size);

    std::vector<int> test_indices(
        indices.begin() + train_size,
        indices.end());

    std::cout << "Train size: "
              << train_indices.size()
              << "\n";

    std::cout << "Test size: "
              << test_indices.size()
              << "\n";

    // ---------------- Model ----------------
    CNN model(num_classes, dataset.channels);

    int total_params = 0;
    for (auto p : model.parameters())
        total_params += p->numel();

    std::cout << "Total Parameters: "
              << total_params
              << "\n";

    // ---------------- Optimizer ----------------
    SGD optimizer(model.parameters(), 0.02f, 0.9f);

    float final_test_acc = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {

        float total_loss = 0;
        float total_acc = 0;
        int batches = 0;

        // ---------------- TRAIN ----------------
        for (int i = 0; i < train_size; i += batch_size) {

            int actual_batch =
                std::min(batch_size,
                         train_size - i);

            Tensor batch_images(
                {actual_batch,
                 dataset.channels,
                 32, 32});

            std::vector<int> batch_labels;

            int img_size =
                dataset.channels * 32 * 32;

            for (int b = 0; b < actual_batch; b++) {

                int idx = train_indices[i + b];

                for (int k = 0; k < img_size; k++)
                    batch_images.data[b * img_size + k] =
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

            for (int k = 0; k < logits.numel(); k++)
                grad.data[k] = logits.grad[k];

            grad = model.fc.backward(grad);

            Tensor reshaped({actual_batch, 64, 2, 2});
            for (int k = 0; k < grad.numel(); k++)
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
            total_acc += compute_accuracy(
                logits,
                batch_labels);

            batches++;
        }

        float train_acc = total_acc / batches;

        // ---------------- TEST ----------------
        float test_acc = 0;
        int test_batches = 0;

        for (int i = 0;
             i < test_indices.size();
             i += batch_size) {

            int actual_batch =
                std::min(batch_size,
                         (int)test_indices.size() - i);

            Tensor batch_images(
                {actual_batch,
                 dataset.channels,
                 32, 32});

            std::vector<int> batch_labels;

            int img_size =
                dataset.channels * 32 * 32;

            for (int b = 0; b < actual_batch; b++) {

                int idx = test_indices[i + b];

                for (int k = 0; k < img_size; k++)
                    batch_images.data[b * img_size + k] =
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

        final_test_acc =
            test_acc / test_batches;

        std::cout << "Epoch "
                  << epoch + 1
                  << " | Loss: "
                  << total_loss / batches
                  << " | Train Acc: "
                  << train_acc * 100
                  << "% | Test Acc: "
                  << final_test_acc * 100
                  << "%\n";
    }

    save_weights(model.parameters(),
                 weights_path);

    return final_test_acc;
}

// ---------------- Evaluate ----------------

float evaluate_model(const std::string &dataset_path,
                     const std::string &weights_path,
                     int num_classes,
                     int batch_size) {

    // ---------------- Dataset Loading ----------------
    ImageDataset dataset(dataset_path);

    CNN model(num_classes, dataset.channels);


    load_weights(model.parameters(),
                 weights_path);

    std::cout << "Weights loaded from "
              << weights_path
              << "\n";


    int total_params = 0;
    for (auto p : model.parameters())
        total_params += p->numel();

    std::cout << "Total Parameters: "
              << total_params
              << "\n";

    float total_acc = 0;
    int batches = 0;

    for (int i = 0;
         i < dataset.size();
         i += batch_size) {

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

    float final_acc =
        total_acc / batches;

    std::cout << "Test Accuracy: "
              << final_acc * 100
              << "%\n";

    return final_acc;
}

// ---------------- Python Module ----------------

PYBIND11_MODULE(customdl, m) {

    m.doc() = "Custom Deep Learning Framework";

    m.def("train", &train_model,
          py::arg("dataset_path"),
          py::arg("num_classes"),
          py::arg("epochs") = 10,
          py::arg("batch_size") = 64,
          py::arg("weights_path") = "model.bin");

    m.def("evaluate", &evaluate_model,
          py::arg("dataset_path"),
          py::arg("weights_path"),
          py::arg("num_classes"),
          py::arg("batch_size") = 64);
}
