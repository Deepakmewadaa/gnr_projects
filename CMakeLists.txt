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


target_include_directories(train
    PRIVATE framework)


target_link_libraries(train
    ${OpenCV_LIBS})
