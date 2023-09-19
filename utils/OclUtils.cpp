//
// Created by 向向泳安 on 2022/3/15.
//

#include "OclUtils.h"

void PrintError(cl_int error);

inline void CheckError(cl_int iStatus, std::string errMsg) {
  if (CL_SUCCESS != iStatus) {
        std::cout << errMsg << std::endl;
        PrintError(iStatus);
		exit(1);
	}
}

inline void noPtrCheck(void *ptr, std::string errMsg) {
    if (NULL == ptr) {
        std::cout << "error: " << errMsg << std::endl;
        exit(0);
    }
}

std::string load_program(const char* filename)
{
    size_t	program_size[1];
    FILE* program_handle = fopen(filename, "rb");
    if (program_handle == nullptr)
        perror("Error opening file\n");

    fseek(program_handle, 0, SEEK_END);
    program_size[0] = ftell(program_handle);
    std::fstream kernelFile(filename);
    std::string content(
            (std::istreambuf_iterator<char>(kernelFile)),
            std::istreambuf_iterator<char>()
    );
    return content;
}

void OclDevice::GetDeviceInfo(std::string Dname) {

}

void OclDevice::SetContext() {
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, NULL);
    noPtrCheck(this->context, "Can not create context");
}

void OclDevice::SetQueue() {
    queue = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, NULL);
    noPtrCheck(queue, "Can not create CommandQueue");
}

void OclDevice::CreateProgramWithSource(std::string filePath) {
    std::string content = load_program(filePath.c_str());
    const char* kernelCharArray = new char[content.size()];
    kernelCharArray = content.c_str();
    program = clCreateProgramWithSource(context, 1, &kernelCharArray, NULL, NULL);
    noPtrCheck(program, "Can not create program");

    cl_int iStatus = 0;
    iStatus = clBuildProgram(this->program, 1, &deviceId, NULL, NULL, NULL);
    if (CL_SUCCESS != iStatus)
    {
        std::cout << "Error: Can not build program" << std::endl;
        char szBuildLog[16384];
        clGetProgramBuildInfo(this->program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);
        std::cout << "Error in Kernel: " << std::endl << szBuildLog;
        exit(0);
    }
}

int OclDevice::SetKernel(std::string kernelName) {
    OclKernel k;
    kernelLists.push_back(k);
    nameMapKernel[kernelName] = kernelLists.size()-1;
    k.kernel = clCreateKernel(program, kernelName.c_str(), nullptr);
    noPtrCheck(k.kernel, "Can not create kernel :" + kernelName);
    return nameMapKernel[kernelName];
}

std::string getPlatformName(cl_platform_id pid) {
  cl_int status;char *value;  size_t sz;
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &sz);
  CheckError(status, "Query for platform name size failed");
  value = (char*) malloc(sz);
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sz, value, NULL);
  CheckError(status, "Query for platform name failed");
  return value;
}

OclUtils::OclUtils(std::string Pname) {
  cl_int iStatus = 0;
  cl_uint num;
  cl_uint uiNumPlatforms = 0;
  CheckError(clGetPlatformIDs(0, nullptr, &uiNumPlatforms),
             "Getting platforms error");
  std::vector<cl_platform_id> pPlatforms(uiNumPlatforms, nullptr);
  CheckError(clGetPlatformIDs(uiNumPlatforms, pPlatforms.data(), nullptr),
             "FATAL : fail to get platfrom Info");
  this->platform = pPlatforms[0]; // choice paltfrom
  std::cout<< "Find platfrom : "<<getPlatformName(platform)<<std::endl;
}

void OclUtils::ShowAllDevice() {
    cl_int iStatus = 0;
    cl_uint	uiNumDevices = 0;
    iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &uiNumDevices);
    std::vector<cl_device_id> devices(uiNumDevices, nullptr);
    iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_ALL, uiNumDevices, devices.data(), nullptr);
    for(auto & d : devices) {
        ShowDeviceInfo(d);
    }
}

void OclUtils::ShowDeviceInfo(cl_device_id Did) {
    char* value;
    size_t      valueSize;
    size_t      maxWorkItemPerGroup;
    cl_uint     maxComputeUnits = 0;
    cl_ulong    maxGlobalMemSize = 0;
    cl_ulong    maxConstantBufferSize = 0;
    cl_ulong    maxLocalMemSize = 0;
    ///print the device name
    clGetDeviceInfo(Did, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(Did, CL_DEVICE_NAME, valueSize, value, NULL);
    std::cout << "=======================================" << std::endl;
    printf("Device Name: %s\n", value);
    free(value);

    /// print parallel compute units(CU)
    clGetDeviceInfo(Did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Parallel compute units: %u\n", maxComputeUnits);

    clGetDeviceInfo(Did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
    printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);

    /// print maxGlobalMemSize
    clGetDeviceInfo(Did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
    printf("maxGlobalMemSize: %lu(MB)\n", maxGlobalMemSize / 1024 / 1024);

    /// print maxConstantBufferSize
    clGetDeviceInfo(Did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    printf("maxConstantBufferSize: %lu(KB)\n", maxConstantBufferSize / 1024);

    /// print maxLocalMemSize
    clGetDeviceInfo(Did, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
    printf("maxLocalMemSize: %lu(KB)\n", maxLocalMemSize / 1024);
    std::cout << "=======================================" << std::endl;
}

cl_ulong OclUtils::getStartEndTime(cl_event event) {
    cl_int status;
    cl_ulong start, end;
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    CheckError(status, "Failed to query event start time");
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    CheckError(status, "Failed to query event end time");
    return end - start;
}

void PrintError(cl_int error) {
    switch(error)
    {
        case -1:
            printf("CL_DEVICE_NOT_FOUND ");
            break;
        case -2:
            printf("CL_DEVICE_NOT_AVAILABLE ");
            break;
        case -3:
            printf("CL_COMPILER_NOT_AVAILABLE ");
            break;
        case -4:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
            break;
        case -5:
            printf("CL_OUT_OF_RESOURCES ");
            break;
        case -6:
            printf("CL_OUT_OF_HOST_MEMORY ");
            break;
        case -7:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
            break;
        case -8:
            printf("CL_MEM_COPY_OVERLAP ");
            break;
        case -9:
            printf("CL_IMAGE_FORMAT_MISMATCH ");
            break;
        case -10:
            printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
            break;
        case -11:
            printf("CL_BUILD_PROGRAM_FAILURE ");
            break;
        case -12:
            printf("CL_MAP_FAILURE ");
            break;
        case -13:
            printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
            break;
        case -14:
            printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
            break;

        case -30:
            printf("CL_INVALID_VALUE ");
            break;
        case -31:
            printf("CL_INVALID_DEVICE_TYPE ");
            break;
        case -32:
            printf("CL_INVALID_PLATFORM ");
            break;
        case -33:
            printf("CL_INVALID_DEVICE ");
            break;
        case -34:
            printf("CL_INVALID_CONTEXT ");
            break;
        case -35:
            printf("CL_INVALID_QUEUE_PROPERTIES ");
            break;
        case -36:
            printf("CL_INVALID_COMMAND_QUEUE ");
            break;
        case -37:
            printf("CL_INVALID_HOST_PTR ");
            break;
        case -38:
            printf("CL_INVALID_MEM_OBJECT ");
            break;
        case -39:
            printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
            break;
        case -40:
            printf("CL_INVALID_IMAGE_SIZE ");
            break;
        case -41:
            printf("CL_INVALID_SAMPLER ");
            break;
        case -42:
            printf("CL_INVALID_BINARY ");
            break;
        case -43:
            printf("CL_INVALID_BUILD_OPTIONS ");
            break;
        case -44:
            printf("CL_INVALID_PROGRAM ");
            break;
        case -45:
            printf("CL_INVALID_PROGRAM_EXECUTABLE ");
            break;
        case -46:
            printf("CL_INVALID_KERNEL_NAME ");
            break;
        case -47:
            printf("CL_INVALID_KERNEL_DEFINITION ");
            break;
        case -48:
            printf("CL_INVALID_KERNEL ");
            break;
        case -49:
            printf("CL_INVALID_ARG_INDEX ");
            break;
        case -50:
            printf("CL_INVALID_ARG_VALUE ");
            break;
        case -51:
            printf("CL_INVALID_ARG_SIZE ");
            break;
        case -52:
            printf("CL_INVALID_KERNEL_ARGS ");
            break;
        case -53:
            printf("CL_INVALID_WORK_DIMENSION ");
            break;
        case -54:
            printf("CL_INVALID_WORK_GROUP_SIZE ");
            break;
        case -55:
            printf("CL_INVALID_WORK_ITEM_SIZE ");
            break;
        case -56:
            printf("CL_INVALID_GLOBAL_OFFSET ");
            break;
        case -57:
            printf("CL_INVALID_EVENT_WAIT_LIST ");
            break;
        case -58:
            printf("CL_INVALID_EVENT ");
            break;
        case -59:
            printf("CL_INVALID_OPERATION ");
            break;
        case -60:
            printf("CL_INVALID_GL_OBJECT ");
            break;
        case -61:
            printf("CL_INVALID_BUFFER_SIZE ");
            break;
        case -62:
            printf("CL_INVALID_MIP_LEVEL ");
            break;
        case -63:
            printf("CL_INVALID_GLOBAL_WORK_SIZE ");
            break;
        default:
            printf("UNRECOGNIZED ERROR CODE (%d)", error);
    }

}


