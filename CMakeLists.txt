cmake_minimum_required(VERSION 3.20)
project(HGC)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
message("OpenCL is find : " ${OpenCL_FOUND})

add_library(graph graph/graph.cpp graph/graph.h)
add_library(env env/opencl.cpp env/opencl.h)
target_link_libraries(env ${OpenCL_LIBRARIES})

add_subdirectory(algo)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} algo)