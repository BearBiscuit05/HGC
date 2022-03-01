#include "algo.h"

#define DEBUG
//#define MERGE_DEBUG

void Algo::setEnv(string filePath)
{
	this->env.setEnv(filePath);
	cout << "load GPU env success" << endl;
}

void Algo::loadGraph(string filePath)
{
	//this->graph.readFile2NodeGraph(filePath,4);
	this->graph.readFile2Graph(filePath);
	cout << "load graph success" << endl;
}

void Algo::Engine_GPU(int partition)
{
	int iter = 0;
	vector<int> mValues(this->MemSpace);
	clock_t start, end, sumClock = 0,subStart;
	start = clock();
	while (this->graph.activeNodeNum > 0) {
#ifdef DEBUG
		clock_t subiter;
		cout << "----------------------" << endl;
		cout << "this is iter : " << iter++ << endl;
		subStart = clock();
		subiter = clock();
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		cout << "divide run time: " << (double)(clock() - subStart) / CLOCKS_PER_SEC << "s" << endl;
		subStart = clock();
		for (auto& g : subGraph) {
			mValues.assign(this->MemSpace, INT_MAX);
			subStart = clock();
			//MSGGenMergeByNode_GPU(g, mValues);
			MSGGenMerge_GPU(g, mValues);
			cout << "Gen run time: " << (double)(clock() - subStart) / CLOCKS_PER_SEC << "s" << endl;
			subStart = clock();
			MSGApply_GPU(g, mValues);
			cout << "Apply run time: " << (double)(clock() - subStart) / CLOCKS_PER_SEC << "s" << endl;
		}
		sumClock += clock() - subiter;
		subStart = clock();
		MergeGraph_GPU(subGraph);
		cout << "mergeGraph run time: " << (double)(clock() - subStart) / CLOCKS_PER_SEC << "s" << endl;
		this->graph.activeNodeNum = GatherActiveNodeNum_GPU(this->graph.vertexActive);
		cout << "Gather run time: " << (double)(clock() - subStart) / CLOCKS_PER_SEC << "s" << endl;
		cout << "------------------------------" << endl;
		cout << "iter run  time: " << (double)(clock() - subiter) / CLOCKS_PER_SEC << "s" << endl;
		cout << "active node number" << this->graph.activeNodeNum << endl;
		cout << "------------------------------" << endl;
#else
		cout << "----------------------" << endl;
		cout << "this is iter : " << iter++ << endl;
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		subStart = clock();
		for (auto& g : subGraph) {
			mValues.assign(this->MemSpace, INT_MAX);
			//MSGGenMergeByNode_GPU(g, mValues);
			MSGGenMerge_GPU(g, mValues);
			MSGApply_GPU(g, mValues);
		}
		MergeGraph_GPU(subGraph);
		this->graph.activeNodeNum = GatherActiveNodeNum_GPU(this->graph.vertexActive);
		cout << "------------------------------" << endl;
		cout << "active node number" << this->graph.activeNodeNum << endl;
#endif // DEBUG
	}
	end = clock();
	cout << "================================" << endl;
	cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "count time: " << (double)sumClock / CLOCKS_PER_SEC << "s" << endl;
	cout << "================================" << endl;
}

void Algo::MergeGraph_GPU(vector<Graph>& subGraph)
{
	size_t globalSize = this->graph.vCount;
	cl_int iStatus = 0;
	size_t dim = 1;
	int index = -1;

	int kernelID = 0;
	if (env.nameMapKernel.find("MergeGraph") == env.nameMapKernel.end()) {
		env.nameMapKernel["MergeGraph"] = env.setKernel("MergeGraph");
		kernelID = env.nameMapKernel["MergeGraph"];
		vector<cl_mem> tmp(4, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->graph.vCount * sizeof(int), nullptr, nullptr);//src
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->graph.vCount * sizeof(int), nullptr, nullptr);//dst
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);

		for (int i = 0; i <= index; i++) {
			if (tmp[i] == nullptr)
				env.noPtrCheck(nullptr, "set mem error");
		}
		env.clMem.push_back(tmp);

		for (int i = 0; i <= index; i++) {
			iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
		}
		env.errorCheck(iStatus, "set kernel agrs fail!");
	}
	else {
		kernelID = env.nameMapKernel["MergeGraph"];
	}
	
	cl_event writeEvent,runEvent,readEvent;
	index = -1;
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->graph.vCount * sizeof(int), &subGraph[0].vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->graph.vCount * sizeof(int), &subGraph[1].vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &subGraph[0].distance[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &subGraph[1].distance[0], 0, nullptr, &writeEvent);
	clWaitForEvents(1, &writeEvent);


	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, &runEvent);
	env.errorCheck(iStatus, "Can not run GenMerge kernel");
	clWaitForEvents(1, &runEvent);


	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, this->MemSpace * sizeof(int), &this->graph.vertexActive[0], 0, NULL, NULL);
	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][2], CL_TRUE, 0, this->MemSpace * sizeof(int), &this->graph.distance[0], 0, NULL, &readEvent);
	clWaitForEvents(1, &runEvent);
	env.errorCheck(iStatus, "Can not reading result buffer");
#ifdef MERGE_DEBUG
	cl_ulong time_start, time_end;
	clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double nanoSeconds = time_end - time_start;
	printf("OpenCl write buffer time is: %0.5f milliseconds \n", nanoSeconds / 1000000.0);
	clGetEventProfilingInfo(runEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(runEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is: %0.5f milliseconds \n", nanoSeconds / 1000000.0);
	clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds = time_end - time_start;
	printf("OpenCl read buffer time is: %0.5f milliseconds \n", nanoSeconds / 1000000.0);
#endif //MERGE_DEBUG

}

void Algo::MSGApply_GPU(Graph& g, vector<int>& mValue)
{
	g.activeNodeNum = 0;
	int kernelID = 0, index = 0;
	const size_t globalSize = this->graph.vCount;
	cl_int iStatus = 0;
	size_t dim = 1;

	if (env.nameMapKernel.find("Apply") == env.nameMapKernel.end()) {
		env.nameMapKernel["Apply"] = env.setKernel("Apply");
		kernelID = env.nameMapKernel["Apply"];
		vector<cl_mem> tmp(3, nullptr);
		index = -1;

		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.vCount * sizeof(int), nullptr, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);

		for (int i = 0; i <= index; i++) {
			if (tmp[i] == nullptr)
				env.noPtrCheck(nullptr, "set mem error");
		}
		env.clMem.push_back(tmp);

		for (int i = 0; i <= index; i++) {
			iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
		}
		env.errorCheck(iStatus, "set kernel agrs fail!");
	}
	else {
		kernelID = env.nameMapKernel["Apply"];
	}

	cl_event startEvt;
	index = -1;
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
	clWaitForEvents(1, &startEvt);

	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not run kernel");

	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, NULL, NULL);
	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][2], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not reading result buffer");
}

int Algo::GatherActiveNodeNum_GPU(vector<int>& activeNodes)
{
	int kernelID = 0, index = 0;
	const size_t localSize = 256;
	int len = activeNodes.size();
	int group = len / localSize;
	const size_t globalSize = len;

	cl_int iStatus = 0;
	size_t dim = 1;
	vector<int> subSum(group, 0);
	if (env.nameMapKernel.find("Gather") == env.nameMapKernel.end())
	{
		env.nameMapKernel["Gather"] = env.setKernel("Gather");
		kernelID = env.nameMapKernel["Gather"];
		vector<cl_mem> tmp(2, nullptr);
		tmp[0] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, globalSize * sizeof(int), nullptr, nullptr);
		tmp[1] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, group * sizeof(int), nullptr, nullptr);
		if (tmp[0] == nullptr || tmp[1] == nullptr)
			env.noPtrCheck(nullptr, "set mem fail");

		env.clMem.push_back(tmp);
		env.errorCheck(clSetKernelArg(env.kernels[kernelID], 0, sizeof(cl_mem), &env.clMem[kernelID][0]), "set arg fail");
		env.errorCheck(clSetKernelArg(env.kernels[kernelID], 1, sizeof(cl_mem), &env.clMem[kernelID][1]), "set arg fail");
		env.errorCheck(clSetKernelArg(env.kernels[kernelID], 2, localSize * sizeof(int), nullptr), "set arg fail");
	}
	else {
		kernelID = env.nameMapKernel["Gather"];
	}

	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, globalSize * sizeof(int), &activeNodes[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, nullptr, nullptr);

	env.errorCheck(clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
		"Can not run kernel");

	env.errorCheck(clEnqueueReadBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, NULL, NULL),
		"Can not reading result buffer");

	int sum = 0;
	for (int i = 0; i < group; ++i) {
		sum += subSum[i];
	}
	return sum;
}