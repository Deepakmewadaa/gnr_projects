#include "dataset.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>

namespace fs = std::filesystem;

ImageDataset::ImageDataset(std::string root_dir) {

    auto start_time =
        std::chrono::high_resolution_clock::now();


    std::vector<std::string> class_names;

    for (auto &entry : fs::directory_iterator(root_dir)) {
        if (entry.is_directory()) {
            class_names.push_back(
                entry.path().filename().string());
        }
    }

    std::sort(class_names.begin(),
              class_names.end());

    int class_index = 0;

    for (auto &name : class_names) {
        class_to_idx[name] = class_index++;
    }

    num_classes = class_index;

    std::cout << "Discovered "
              << num_classes
              << " classes.\n";


    size_t total_files = 0;

    for (auto &name : class_names) {

        fs::path class_path =
            fs::path(root_dir) / name;

        for (auto &file :
             fs::directory_iterator(class_path)) {

            if (file.is_regular_file() &&
                file.path().extension() == ".png")
                total_files++;
        }
    }

    images.reserve(total_files);
    labels.reserve(total_files);


    std::vector<std::thread> threads;
    std::mutex mtx;

    for (auto &name : class_names) {

        fs::path class_path =
            fs::path(root_dir) / name;

        int label = class_to_idx[name];

        threads.emplace_back([&, class_path, label]() {

            for (auto &img_file :
                 fs::directory_iterator(class_path)) {

                if (!img_file.is_regular_file())
                    continue;

                if (img_file.path().extension() != ".png")
                    continue;

                cv::Mat img =
                    cv::imread(img_file.path().string(),
                                cv::IMREAD_UNCHANGED);


                if (img.empty())
                    continue;

                cv::resize(img, img,
                           cv::Size(32, 32));
                
             
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (images.empty()) {
                    channels = img.channels();
                }
            }


                Tensor t({channels, 32, 32});

                if (channels == 1) {

                    for (int i = 0; i < 32; i++)
                    for (int j = 0; j < 32; j++) {

                        float val =
                            img.at<uchar>(i,j) / 255.0f;

                        val = (val - 0.5f) / 0.5f;

                        t.data[i*32 + j] = val;
                    }
                }
                else { 

                    for (int c = 0; c < 3; c++)
                    for (int i = 0; i < 32; i++)
                    for (int j = 0; j < 32; j++) {

                        float val =
                            img.at<cv::Vec3b>(i,j)[c] / 255.0f;

                        val = (val - 0.5f) / 0.5f;

                        int index =
                            c*32*32 + i*32 + j;

                        t.data[index] = val;
                    }
                }


                std::lock_guard<std::mutex> lock(mtx);

                images.push_back(t);
                labels.push_back(label);
            }
        });
    }

    for (auto &th : threads)
        th.join();

    auto end_time =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff =
        end_time - start_time;

    std::cout << "Dataset loading time: "
              << diff.count()
              << " seconds\n";

    std::cout << "Total samples: "
              << images.size() << "\n";
}


int ImageDataset::size() const {
    return images.size();
}


void ImageDataset::get_batch(int start,
                             int batch_size,
                             Tensor &batch_images,
                             std::vector<int> &batch_labels) {

    int end = std::min(start + batch_size,
                       (int)images.size());

    int actual_batch = end - start;

    batch_images =
        Tensor({actual_batch, channels, 32, 32});

    batch_labels.clear();

    for (int b = 0; b < actual_batch; b++) {

        int img_index = start + b;

        int img_size = channels * 32 * 32;

        for (int i = 0; i < img_size; i++) {

            batch_images.data[b*img_size + i] =
                images[img_index].data[i];
        }


        batch_labels.push_back(labels[img_index]);
    }
}
