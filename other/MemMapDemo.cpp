#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <malloc.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if _WIN32 || WIN32
#include <windows.h>
#include <time.h>
#include <direct.h>
#include <io.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define LENGTH       10
#define KERNEL_FUNC  "addVec"

//#define MEM_MAP    //映射内存对象

int main(void)
{
	cl_platform_id* platformALL = NULL;
	cl_uint ret_num_platforms;

	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;

	cl_context context = NULL;
	cl_command_queue command_queue = NULL;

	cl_program program = NULL;
	cl_kernel kernel = NULL;

	cl_int ret, err;

	const char* kernel_src_str = "\n" \
		"__kernel void addVec(__global int *dataSrc){"\
		"float16 aa; int idx = get_global_id(0);"\
		"if (idx<10){"\
		"	dataSrc[idx] += 10;"\
		"}}";

	//Step1:获取平台列表
	err = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (ret_num_platforms < 1)
	{
		printf("Error: Getting Platforms; err = %d,numPlatforms = %d !", err, ret_num_platforms);
	}
	printf("Num of Getting Platforms = %d!\n", ret_num_platforms);

	platformALL = (cl_platform_id*)alloca(sizeof(cl_platform_id) * ret_num_platforms);
	ret = clGetPlatformIDs(ret_num_platforms, platformALL, &ret_num_platforms);

	//Step2:获取指定设备，platformALL[0],platformALL[1]...
	//带有独显的PC,选择intel核显或独显
	ret = clGetDeviceIDs(platformALL[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	char  nameDevice[64];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(nameDevice), &nameDevice, NULL);
	printf("Device Name: %s\n", nameDevice);

	//Step3:创建上下文
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret < 0)
	{
		printf("clCreateContext Fail,ret =%d\n", ret);
		exit(1);
	}

	//Step4:创建命令队列
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (ret < 0)
	{
		printf("clCreateCommandQueue Fail,ret =%d\n", ret);
		exit(1);
	}

	//Step5:创建程序
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_src_str, NULL, &ret);
	if (ret < 0)
	{
		perror("clCreateProgramWithSource Fail\n");
		exit(1);
	}
	//Step6:编译程序
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret < 0)
	{
		perror("clBuildProgram Fail\n");
		exit(1);
	}

	//Step7:创建内核
	kernel = clCreateKernel(program, KERNEL_FUNC, &ret);
	if (ret < 0)
	{
		perror("clCreateKernel Fail\n");
		exit(1);
	}

	printf("GPU openCL init Finish\n");

	cl_mem clDataSrcBuf;
	int dataSrcHost[LENGTH] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	//创建缓存对象
#ifdef MEM_MAP
	clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * LENGTH, NULL, &err);
#else
	clDataSrcBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * LENGTH, dataSrcHost, &err);
#endif
	if (err < 0)
	{
		printf("clCreateBuffer imgSrcBuf Fail,err=%d\n", err);
		exit(1);
	}

#ifdef MEM_MAP
	cl_int* bufferMap = (cl_int*)clEnqueueMapBuffer(command_queue, clDataSrcBuf, CL_TRUE, CL_MAP_WRITE,
		0, LENGTH * sizeof(cl_int), 0, NULL, NULL, NULL);
	memcpy(bufferMap, dataSrcHost, 10 * sizeof(int));
#endif

	//设置内核参数
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clDataSrcBuf);
	if (err < 0)
	{
		perror("clSetKernelArg imgDstBuf Fail\n");
		exit(1);
	}

	printf("GPU openCL Create and set Buffter Finish\n");
	cl_uint work_dim = 1;
	size_t  global_item_size = LENGTH;

	err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, &global_item_size, NULL, 0, NULL, NULL);
	clFinish(command_queue);
	if (err < 0)
	{
		printf("err:%d\n", err);
		perror("clEnqueueNDRangeKernel Fail\n");

	}

#ifndef MEM_MAP
	err = clEnqueueReadBuffer(command_queue, clDataSrcBuf, CL_TRUE, 0, (4 * LENGTH), dataSrcHost, 0, NULL, NULL);
	if (err < 0)
	{
		printf("err:%d\n", err);
		perror("Read buffer command Fail\n");

	}
#endif

	//print result
	for (int i = 0; i < LENGTH; i++)
	{
#ifdef MEM_MAP
		printf("dataSrcHost[%d]:%d\n", i, bufferMap[i]);
#else
		printf("dataSrcHost[%d]:%d\n", i, dataSrcHost[i]);
#endif
	}

#ifdef MEM_MAP
	err = clEnqueueUnmapMemObject(command_queue, clDataSrcBuf, bufferMap, 0, NULL, NULL);
#endif

	/* OpenCL Object Finalization */
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	return 0;
}