cmake_minimum_required(VERSION 2.8)

project(flow)

find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(
	main.cpp
)

target_link_libraries(planeProjection ${OpenCV_LIBS})
