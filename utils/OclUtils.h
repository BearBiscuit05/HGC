#ifndef OPENCL_WRAPPER_OCLUTILS_H
#define OPENCL_WRAPPER_OCLUTILS_H
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
//                  -> args
// device -> kernels
class OclKernel;

class OclDevice {
public:
    void GetDeviceInfo(std::string Dname = "");
    void SetContext();
    void SetQueue();
    void CreateProgramWithSource(std::string filePath);
    int SetKernel(std::string kernelName);

    std::string deviceName = "";
    cl_device_id deviceId = nullptr;
    cl_command_queue queue = nullptr;
    cl_context context = nullptr;
    cl_program program = nullptr;
    std::vector<OclKernel> kernelLists;
    std::unordered_map<std::string, int> nameMapKernel;
};

class OclKernel {
public:
    void SetKernelArgs();
    void WriteData();
    void Run();
    void ReadData();

    cl_kernel kernel;
    cl_device_id deviceId;
    std::string kernelName = "";

    int dim = 0;
    std::vector<int> globalSize;
    std::vector<int> localSize;
    std::vector<cl_mem> clMemLists;
};

class OclUtils {
public:
    // TODO : init platform detect
    OclUtils(std::string Pname = "");
    void ShowAllDevice();
    void ShowDeviceInfo(cl_device_id Did);
    void GetPlatfromInfo(std::string Pname = "");
    cl_ulong getStartEndTime(cl_event event);


    cl_platform_id platform = nullptr;
};


#endif //OPENCL_WRAPPER_OCLUTILS_H
