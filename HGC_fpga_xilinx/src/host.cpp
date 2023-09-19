#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include "xcl2.hpp"

//#define MAX_VERTEX_IN_ONE_PARTITION (524288+1)
//#define MAX_EDGE_IN_ONE_PARTITION (524288*2)
#define MAX_VERTEX_IN_ONE_PARTITION (4096+1)
#define MAX_EDGE_IN_ONE_PARTITION (4096*4)

int main(int argc, char** argv) {
    std::vector<int,aligned_allocator<int>> src(MAX_EDGE_IN_ONE_PARTITION,8);
    std::vector<int,aligned_allocator<int>> dst(MAX_EDGE_IN_ONE_PARTITION,8);
    std::vector<int,aligned_allocator<int>> vertexValue(MAX_VERTEX_IN_ONE_PARTITION,0);
    std::vector<int,aligned_allocator<int>> tmpValue(MAX_VERTEX_IN_ONE_PARTITION,0);
    src[0] = 0,src[1] =1,src[2] = 2,src[3] = 2,src[4] = 3,src[5] = 3,src[6] = 4,src[7] = 5;
    dst[0] = 1,dst[1] =2,dst[2] = 1,dst[3] = 5,dst[4] = 2,dst[5] = 0,dst[6] = 3,dst[7] = 6;
    vertexValue[0] = 1;
    int edgeNum = MAX_EDGE_IN_ONE_PARTITION;
    int activeNodeNum = 1;
    int vertex4pe = MAX_VERTEX_IN_ONE_PARTITION;
    int vertexBegin = 0;

    //get device
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Device device;
    bool found_device = false;
    cl_int err;

    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

            //Traversing All Devices of Xilinx Platform
            for (size_t j = 0 ; j < devices.size() ; j++){
                device = devices[j];
                std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
                std::cout << deviceName << std::endl;
                if (deviceName == "xilinx_aliyun-f3_dynamic_5_0"){
                    found_device = true;
                    break;
                }
            }
        }
    }
    // if (found_device == false){
    //    std::cout << "Error: Unable to find Target Device " 
    //        << "xilinx_aliyun-f3_dynamic_5_0" << std::endl;
    //    return EXIT_FAILURE; 
    //}

    
    // create context and queue
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    std::string binaryFile = xcl::find_binary_file(device_name, "gs_top");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    
    while(activeNodeNum > 0){
        OCL_CHECK(err, cl::Kernel krnl(program, "gs_top", &err));

        std::cout << "create buffer" << "'\n";
        cl::Buffer buffer_a(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,  MAX_EDGE_IN_ONE_PARTITION * sizeof(int), src.data());
        cl::Buffer buffer_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,  MAX_VERTEX_IN_ONE_PARTITION * sizeof(int), vertexValue.data());
        cl::Buffer buffer_c(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,  MAX_EDGE_IN_ONE_PARTITION * sizeof(int), dst.data());
        cl::Buffer buffer_d(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,  MAX_VERTEX_IN_ONE_PARTITION * sizeof(int), tmpValue.data());
        
        std::cout << "set args" << "'\n";
        OCL_CHECK(err, err = krnl.setArg(0, buffer_a));
        OCL_CHECK(err, err = krnl.setArg(1, buffer_b));
        OCL_CHECK(err, err = krnl.setArg(2, buffer_c));
        OCL_CHECK(err, err = krnl.setArg(3, buffer_d));
        OCL_CHECK(err, err = krnl.setArg(4, edgeNum));
        OCL_CHECK(err, err = krnl.setArg(5, vertexBegin));
        OCL_CHECK(err, err = krnl.setArg(6, vertex4pe));
        OCL_CHECK(err, err = krnl.setArg(7, activeNodeNum));

        std::cout << "migrate mem" << "'\n";
        q.enqueueMigrateMemObjects({buffer_a,buffer_b,buffer_c,buffer_d},0/* 0 means from host*/);
        
        std::cout << "run kernel" << "'\n";
        cl::Event event;
        OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event));
        OCL_CHECK(err, err = event.wait());
        std::cout << "kernel finish" << "'\n";
        q.enqueueMigrateMemObjects({buffer_d},CL_MIGRATE_MEM_OBJECT_HOST);
        std::cout << "mem read finish" << "'\n";
        q.finish();
        std::cout << "all finish" << "'\n";
        for(int i = 0 ; i < 8 ; i++) {
            std::cout<< "src " << i <<"'s value is :" << tmpValue[i] << std::endl;
        }

        std::cout << "activeNodeNum :" << tmpValue[MAX_VERTEX_IN_ONE_PARTITION-1] << std::endl; 
        activeNodeNum = tmpValue[MAX_VERTEX_IN_ONE_PARTITION-1];
        tmpValue[MAX_VERTEX_IN_ONE_PARTITION-1] = 0;
        for(int i = 0 ; i < MAX_VERTEX_IN_ONE_PARTITION ; i++)
        {
            vertexValue[i] = tmpValue[i];
        }
    }
    
    return 0;
}
