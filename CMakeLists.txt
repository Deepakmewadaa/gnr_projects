cmake_minimum_required(VERSION 3.21)

cmake_minimum_required(VERSION 3.15)
project(CustomDL)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB FRAMEWORK_SOURCES
     framework/*.cpp)

# -------------------------
# Train executable
# -------------------------
add_executable(train
    train.cpp
    ${FRAMEWORK_SOURCES}
)

target_include_directories(train PRIVATE framework ${OpenCV_INCLUDE_DIRS})
target_link_libraries(train PRIVATE ${OpenCV_LIBS})

# -------------------------
# Evaluate executable
# -------------------------
add_executable(evaluate
    evaluate.cpp
    ${FRAMEWORK_SOURCES}
)

target_include_directories(evaluate PRIVATE framework ${OpenCV_INCLUDE_DIRS})
target_link_libraries(evaluate PRIVATE ${OpenCV_LIBS})

# -------------------------
# Python module
# -------------------------
pybind11_add_module(customdl
    bindings.cpp
    ${FRAMEWORK_SOURCES}
)

target_include_directories(customdl PRIVATE framework ${OpenCV_INCLUDE_DIRS})
target_link_libraries(customdl PRIVATE ${OpenCV_LIBS})
