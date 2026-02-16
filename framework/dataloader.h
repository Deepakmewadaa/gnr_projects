#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "tensor.h"

using namespace std;
namespace fs = std::filesystem;

struct Dataset {
    vector<string> paths;
    vector<int> labels;
    int channels;

    Dataset(string root, int ch) : channels(ch) {
        int label = 0;
        for (auto &dir : fs::directory_iterator(root)) {
            for (auto &file : fs::directory_iterator(dir.path())) {
                paths.push_back(file.path().string());
                labels.push_back(label);
            }
            label++;
        }
    }

    pair<Tensor,int> get(int idx) {
        cv::Mat img = cv::imread(paths[idx],
                     channels==1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

        cv::resize(img, img, cv::Size(32,32));
        img.convertTo(img, CV_32F, 1.0/255);

        Tensor t({channels,32,32});
        memcpy(t.data.data(), img.data,
               32*32*channels*sizeof(float));

        return {t, labels[idx]};
    }

    int size() { return paths.size(); }
};
