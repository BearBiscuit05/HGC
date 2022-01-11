#include "opencl.h"

void printError(cl_int error)
{
    switch (error)
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

void Env::errorCheck(cl_int iStatus, string errMsg) {
	if (CL_SUCCESS != iStatus) {
        cout << errMsg << endl;
        printError(iStatus);
		exit(0);
	}
}

void Env::noPtrCheck(void* ptr, string errMsg) {
	if (NULL == ptr) {
		cout << "error: " << errMsg << endl;
		exit(0);
	}
}

string load_program(const char* filename)
{
    size_t	program_size[1];
    FILE* program_handle = fopen(filename, "rb");
    if (program_handle == NULL)
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

void Env::setEnv(string filePath)
{
    setPlatform();
    setDevice();
    setContext();
    setQueue();
    setProgram(filePath);
    buildProgram();
}

void Env::setPlatform()
{
    cl_int iStatus = 0;
    cl_uint num;
    cl_uint	uiNumPlatforms = 0;
    errorCheck(clGetPlatformIDs(0, nullptr, &uiNumPlatforms), "Getting platforms error");
    auto* pPlatforms = (cl_platform_id*)malloc(uiNumPlatforms * sizeof(cl_platform_id));
    iStatus = clGetPlatformIDs(uiNumPlatforms, pPlatforms, nullptr);
    this->platform = pPlatforms[0];
    free(pPlatforms);
}

void Env::setDevice()
{
    cl_int iStatus = 0;
    cl_uint	uiNumDevices = 0;
    cl_device_id* pDevices = nullptr;
    iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    if (0 == uiNumDevices)
    {
        cout << "No GPU device available." << endl;
        cout << "Choose CPU as default device." << endl;
        iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
        pDevices = (cl_device_id*)malloc(uiNumDevices * sizeof(cl_device_id));
        iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_CPU, uiNumDevices, pDevices, NULL);
    }
    else
    {
        pDevices = (cl_device_id*)malloc(uiNumDevices * sizeof(cl_device_id));
        iStatus = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, uiNumDevices, pDevices, NULL);
    }
    this->device = pDevices[0];
}

void Env::setContext()
{
    this->context = clCreateContext(NULL, 1, &this->device, NULL, NULL, NULL);
    noPtrCheck(this->context, "Can not create context");
}

void Env::setQueue()
{
    this->queue = clCreateCommandQueue(this->context, this->device, 0, NULL);
    noPtrCheck(this->queue, "Can not create CommandQueue");
}

void Env::setProgram(string filePath)
{
    string content = load_program(filePath.c_str());
    const char* kernelCharArray = new char[content.size()];
    kernelCharArray = content.c_str();
    this->program = clCreateProgramWithSource(this->context, 1, &kernelCharArray, NULL, NULL);
    noPtrCheck(this->program, "Can not create program");
}

int Env::setKernel(string kernelName)
{
    cl_kernel kernel = clCreateKernel(this->program, kernelName.c_str(), nullptr);
    noPtrCheck(kernel, "Can not create kernel");
    this->kernels.push_back(kernel);
    return kernels.size()-1;
}

void Env::showDeviceInfo()
{
    setPlatform();
    setDevice();
    char* value;
    size_t      valueSize;
    size_t      maxWorkItemPerGroup;
    cl_uint     maxComputeUnits = 0;
    cl_ulong    maxGlobalMemSize = 0;
    cl_ulong    maxConstantBufferSize = 0;
    cl_ulong    maxLocalMemSize = 0;
    ///print the device name
    clGetDeviceInfo(this->device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(this->device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device Name: %s\n", value);
    free(value);

    /// print parallel compute units(CU)
    clGetDeviceInfo(this->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Parallel compute units: %u\n", maxComputeUnits);
    
    clGetDeviceInfo(this->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
    printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);

    /// print maxGlobalMemSize
    clGetDeviceInfo(this->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
    printf("maxGlobalMemSize: %lu(MB)\n", maxGlobalMemSize / 1024 / 1024);

    /// print maxConstantBufferSize
    clGetDeviceInfo(this->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    printf("maxConstantBufferSize: %lu(KB)\n", maxConstantBufferSize / 1024);

    /// print maxLocalMemSize
    clGetDeviceInfo(this->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
    printf("maxLocalMemSize: %lu(KB)\n", maxLocalMemSize / 1024);
}

void Env::buildProgram()
{
    cl_int iStatus = 0;
    iStatus = clBuildProgram(this->program, 1, &this->device, NULL, NULL, NULL);
    if (CL_SUCCESS != iStatus)
    {
        cout << "Error: Can not build program" << endl;
        char szBuildLog[16384];
        clGetProgramBuildInfo(this->program, this->device, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);
        cout << "Error in Kernel: " << endl << szBuildLog;
        clReleaseProgram(this->program);
        exit(0);
    }
}

