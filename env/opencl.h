#define _CRT_SECURE_NO_DEPRECATE
#pragma once
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
using namespace std;

class Env
{
public:
	void setEnv(string filePath);
	void setPlatform();
	void setDevice();
	void setContext();
	void setQueue();
	void setProgram(string filePath);
	void buildProgram();
	int setKernel(string kernelName);
	void showDeviceInfo();

	void errorCheck(cl_int iStatus, string errMsg);
	void noPtrCheck(void* ptr, string errMsg);

	bool init = false;
	int memNum = 0;
	cl_platform_id platform = nullptr;
	cl_device_id device = nullptr;
	cl_program program = nullptr;
	cl_context context = nullptr;
	vector<cl_kernel> kernels;
	cl_command_queue queue = nullptr;
	vector<vector<cl_mem>> clMem;
	unordered_map<string, int> nameMapKernel;
};

