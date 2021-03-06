#include <iostream>
#define _CRT_SECURE_NO_WARNINGS
/* 执意使用老版本、非安全性的函数，可以使用 _CRT_SECURE_NO_WARNINGS 标记来忽略这些警告问题*/
#define PROGRAM_FILE "env/kernel/reduction_complete.cl"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 设置输出数组的大小，在预编译阶段会申请一部分内存，申请内存的大小若超过限制(一般为1M)则会报错*/

/* 忽略No.4996警告，即OpenCL引起的版本冲突（许多指令由于安全性等已被禁用或已被更新为其他指令）*/
#pragma warning(disable : 4996)

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
/* 发现可用平台下的GPU或CPU设备*/
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	/* 识别平台*/
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Access a device */
	/* 使用一个设备*/
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

/* Create program from a file and compile it */
/* 从一个文件创建项目并进行编译*/
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE* program_handle;
	char* program_buffer, * program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "rb");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

using namespace std;

int main() {
	int ARRAY_SIZE = 1048576 * 16;
	int NUM_KERNELS = 2;
	int LOCAL_SIZE = 128;
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel kernel[2];
	cl_command_queue queue;
	cl_event prof_event;
	cl_int i, j, err;
	size_t local_size, global_size;
	char kernel_names[2][20] =
	{ "reduction_scalar", "reduction_vector" };

	float data[1048576 * 16];
	float sum, actual_sum, * scalar_sum, * vector_sum;
	cl_mem data_buffer, scalar_sum_buffer, vector_sum_buffer;
	cl_int num_groups;
	cl_ulong time_start, time_end, total_time;
	clock_t start, end;
	for (i = 0; i < ARRAY_SIZE; i++) {
		data[i] = 1.0f * i;
	}

	device = create_device();
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(local_size), &local_size, NULL);
	local_size = LOCAL_SIZE;
	if (err < 0) {
		perror("Couldn't obtain device information");
		exit(1);
	}

	num_groups = ARRAY_SIZE / local_size;
	scalar_sum = (float*)malloc(num_groups * sizeof(float));
	vector_sum = (float*)malloc(num_groups / 4 * sizeof(float));
	for (i = 0; i < num_groups; i++) {
		scalar_sum[i] = 0.0f;
	}
	for (i = 0; i < num_groups / 4; i++) {
		vector_sum[i] = 0.0f;
	}

	/* Create a context */
	/* 创建一个上下文*/
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	/* Build program */
	/* 构建程序*/
	program = build_program(context, device, PROGRAM_FILE);

	/* Create data buffer */
	/* 创建数据buffer缓存，OpenCL一共可以创建Buffer和Image两种内存对象类型，实际应用中具体用途有所区别*/
	data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), data, &err);
	scalar_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), scalar_sum, &err);
	vector_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), vector_sum, &err);
	if (err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	};

	/* Create a command queue */
	/* 创建一个命令队列*/
	queue = clCreateCommandQueue(context, device,
		CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	for (i = 0; i < NUM_KERNELS; i++) {

		/* Create a kernel */
		/* 创建内核*/
		kernel[i] = clCreateKernel(program, kernel_names[i], &err);
		if (err < 0) {
			perror("Couldn't create a kernel");
			exit(1);
		};

		/* Create kernel arguments */
		/* 多个kernel程序进行内核参数设定，i=1为kernel系数*/
		err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &data_buffer);
		if (i == 0) {
			global_size = ARRAY_SIZE;
			err |= clSetKernelArg(kernel[i], 1, local_size * sizeof(float), NULL);
			err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &scalar_sum_buffer);
		}
		else {
			global_size = ARRAY_SIZE / 4;
			err |= clSetKernelArg(kernel[i], 1, local_size * 4 * sizeof(float), NULL);
			err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &vector_sum_buffer);
		}
		if (err < 0) {
			perror("Couldn't create a kernel argument");
			exit(1);
		}

		/* Enqueue kernel */
		/* 将命令入列，若依次执行多个kernel函数，则可以设置一个循环循环入列执行操作，clFinish是否进入循环待验证*/
		err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, &global_size,
			&local_size, 0, NULL, &prof_event);
		if (err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}

		/* Finish processing the queue and get profiling information */
		/* 完成命令队列的处理并得到概述信息*/
		clFinish(queue);

		/*通过clGetEventProfilingInfo得到事件发生的时间*/
		clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
			sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
			sizeof(time_end), &time_end, NULL);
		/*计算每个过程执行时间*/
		total_time = time_end - time_start;

		/* Read the result */
		/* 读取结果*/

		if (i == 0) {
			/*将标量传入kernel计算*/
			err = clEnqueueReadBuffer(queue, scalar_sum_buffer, CL_TRUE, 0,
				num_groups * sizeof(float), scalar_sum, 0, NULL, NULL);
			if (err < 0) {
				perror("Couldn't read the buffer");
				exit(1);
			}
			sum = 0.0f;
			for (j = 0; j < num_groups; j++) {
				sum += scalar_sum[j];
			}
		}
		else {
			/*将向量传入kernel计算*/
			err = clEnqueueReadBuffer(queue, vector_sum_buffer, CL_TRUE, 0,
				num_groups / 4 * sizeof(float), vector_sum, 0, NULL, NULL);
			if (err < 0) {
				perror("Couldn't read the buffer");
				exit(1);
			}
			sum = 0.0f;
			for (j = 0; j < num_groups / 4; j++) {
				sum += vector_sum[j];
			}
		}

		/* Check result */
		/* 校验运算结果*/
		printf("%s: ", kernel_names[i]);
		actual_sum = 1.0f * ARRAY_SIZE / 2 * (ARRAY_SIZE - 1);
		if (fabs(sum - actual_sum) > 0.01 * fabs(sum))
			printf("Check failed.\n");
		else
			printf("Check passed.\n");
		//std::cout << "gpu Total time =  " << total_time<<"*1E-6 ms"<< std::endl;
		printf("gpu: %f ms\n", total_time * 1e-6);
		/* Deallocate event */
		/* 释放事件——为什么要每次释放事件? 实时监测每次运算中的报错，prof_event作为参数输入不同cl函数可以输出不同信息*/
		clReleaseEvent(prof_event);
	}

	float temp = 0.0;
	start = clock();//star
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		temp += data[1];
	}
	end = clock();//end
	printf("cpu: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
	/* Deallocate resources */
	/* 释放资源*/
	free(scalar_sum);
	free(vector_sum);
	for (i = 0; i < NUM_KERNELS; i++) {
		clReleaseKernel(kernel[i]);
	}
	/* 在循环执行的最后释放所有的内存对象，这些内存对象可以循环使用不需要中途释放重新建立，否则太影响效率*/
	clReleaseMemObject(scalar_sum_buffer);
	clReleaseMemObject(vector_sum_buffer);
	clReleaseMemObject(data_buffer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}

