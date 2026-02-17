#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "framework/dataset.h"
#include "framework/conv2d.h"
#include "framework/relu.h"
#include "framework/maxpool.h"
#include "framework/linear.h"
#include "framework/loss.h"
#include "framework/optimizer.h"
#include "framework/weight_io.h"

namespace py = pybind11;

// Forward declarations
float train_model(std::string dataset_path, int num_classes);
float evaluate_model(std::string dataset_path, std::string weights_path);

PYBIND11_MODULE(customdl, m) {

    m.doc() = "Custom Deep Learning Framework";

    m.def("train", &train_model,
          "Train model",
          py::arg("dataset_path"),
          py::arg("num_classes"));

    m.def("evaluate", &evaluate_model,
          "Evaluate model",
          py::arg("dataset_path"),
          py::arg("weights_path"));
}
