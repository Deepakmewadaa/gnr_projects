#include "weight_io.h"
#include <fstream>
#include <iostream>

void save_weights(std::vector<Tensor*> params,
                  const std::string &filename) {

    std::ofstream out(filename,
                      std::ios::binary);

    if (!out.is_open()) {
        std::cout << "Error opening file for saving.\n";
        return;
    }

    for (auto p : params) {

        int size = p->numel();

        out.write(
            reinterpret_cast<char*>(p->data.data()),
            size * sizeof(float));
    }

    out.close();

    std::cout << "Model saved to "
              << filename << "\n";
}


void load_weights(std::vector<Tensor*> params,
                  const std::string &filename) {

    std::ifstream in(filename,
                     std::ios::binary);

    if (!in.is_open()) {
        std::cout << "Error opening file for loading.\n";
        return;
    }

    for (auto p : params) {

        int size = p->numel();

        in.read(
            reinterpret_cast<char*>(p->data.data()),
            size * sizeof(float));
    }

    in.close();

    std::cout << "Model loaded from "
              << filename << "\n";
}
