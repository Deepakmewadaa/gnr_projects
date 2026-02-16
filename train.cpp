#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <random>
#include "framework/dataset.h"
#include "framework/conv2d.h"
#include "framework/relu.h"
#include "framework/maxpool.h"
#include "framework/linear.h"
#include "framework/loss.h"
#include "framework/optimizer.h"


std::map<std::string,std::string>
load_config(std::string path) {

    std::ifstream file(path);
    std::map<std::string,std::string> config;

    std::string line;

    while (std::getline(file,line)) {

        std::stringstream ss(line);
        std::string key,value;

        if (std::getline(ss,key,'=') &&
            std::getline(ss,value)) {

            config[key] = value;
        }
    }

    return config;
}


float compute_accuracy(Tensor &logits,
                       std::vector<int> &labels) {

    int batch = logits.shape[0];
    int classes = logits.shape[1];

    int correct = 0;

    for (int b=0;b<batch;b++) {

        int best = 0;
        float best_val = -1e9;

        for (int c=0;c<classes;c++) {

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


void save_weights(std::vector<Tensor*> params,
                  std::string filename) {

    std::ofstream out(filename,
                      std::ios::binary);

    for (auto p : params) {
        out.write(
            reinterpret_cast<char*>(p->data.data()),
            p->numel()*sizeof(float));
    }

    out.close();
}

void load_weights(std::vector<Tensor*> params,
                  std::string filename) {

    std::ifstream in(filename,
                     std::ios::binary);

    for (auto p : params) {
        in.read(
            reinterpret_cast<char*>(p->data.data()),
            p->numel()*sizeof(float));
    }

    in.close();
}


class CNN {
public:

    Conv2D conv1;
    Conv2D conv2;
    Conv2D conv3;

    ReLU relu1, relu2, relu3;
    MaxPool2D pool1, pool2, pool3;

    Linear fc;

    CNN(int num_classes,
        int in_channels,
        int c1,
        int c2,
        int c3)
    :
      conv1(in_channels,c1,3),
      conv2(c1,c2,3),
      conv3(c2,c3,3),
      pool1(2),
      pool2(2),
      pool3(2),
      fc(c3*2*2, num_classes) {}

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

        Tensor flat({N, fc.in_features});

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


int main(int argc, char* argv[]) {

    if (argc < 4) {

        std::cout << "Usage:\n";
        std::cout << "Train: ./train <dataset_path> <config_path> train\n";
        std::cout << "Eval : ./train <dataset_path> <config_path> eval <weights>\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::string config_path  = argv[2];
    std::string mode         = argv[3];

    std::string weights_path;

    if (mode == "eval") {
        if (argc < 5) {
            std::cout << "Provide weights path for eval\n";
            return 1;
        }
        weights_path = argv[4];
    }


    ImageDataset dataset(dataset_path);

    auto config = load_config(config_path);

    int num_classes =
        std::stoi(config["num_classes"]);

    int epochs =
        std::stoi(config["epochs"]);

    int batch_size =
        std::stoi(config["batch_size"]);

    float lr =
        std::stof(config["learning_rate"]);

    int c1 =
        std::stoi(config["conv1_out"]);
    int c2 =
        std::stoi(config["conv2_out"]);
    int c3 =
        std::stoi(config["conv3_out"]);

    CNN model(
        num_classes,
        dataset.channels,
        c1,c2,c3
    );


    if (mode == "train") {

        SGD optimizer(model.parameters(),
                      lr, 0.9f);

        for (int epoch=0;epoch<epochs;epoch++) {

            float total_loss = 0;
            float total_acc  = 0;
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

                int N = batch_images.shape[0];

                Tensor reshaped({N,c3,2,2});

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

            std::cout << "Epoch "
                      << epoch+1
                      << " | Loss: "
                      << total_loss/batches
                      << " | Acc: "
                      << (total_acc/batches)*100
                      << "%\n";
        }

        save_weights(model.parameters(),
                     "model.bin");

        std::cout << "Training Complete. "
                  << "Weights saved to model.bin\n";
    }

 
    else if (mode == "eval") {

        load_weights(model.parameters(),
                     weights_path);

        float total_acc = 0;
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
    }

    return 0;
}
