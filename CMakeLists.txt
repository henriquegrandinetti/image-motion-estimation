cmake_minimum_required(VERSION 2.8)

project(motionEstimation)

find_package(OpenCV REQUIRED)

add_executable(motionEstimation main.cpp)

target_link_libraries(motionEstimation ${OpenCV_LIBS})
