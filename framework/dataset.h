#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"
#include <vector>
#include <string>
#include <map>

class ImageDataset {
public:
    std::vector<Tensor> images;
    std::vector<int> labels;
    std::map<std::string, int> class_to_idx;
    int channels;
    int num_classes;

    ImageDataset(std::string root_dir);

    int size() const;

    void get_batch(int start,
                   int batch_size,
                   Tensor &batch_images,
                   std::vector<int> &batch_labels);
};

#endif
