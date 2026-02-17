cmake_minimum_required(VERSION 3.10)
project(CustomDL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(OpenCV REQUIRED)

file(GLOB FRAMEWORK_SOURCES
     framework/*.cpp)

add_executable(train
               train.cpp
               ${FRAMEWORK_SOURCES})

add_executable(evaluate
               evaluate.cpp
               ${FRAMEWORK_SOURCES})

target_include_directories(train PRIVATE framework ${OpenCV_INCLUDE_DIRS})
target_include_directories(evaluate PRIVATE framework ${OpenCV_INCLUDE_DIRS})

target_link_libraries(train PRIVATE ${OpenCV_LIBS})
target_link_libraries(evaluate PRIVATE ${OpenCV_LIBS})
