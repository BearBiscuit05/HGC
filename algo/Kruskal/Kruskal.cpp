#include "Kruskal.h"

Kruskal::Kruskal(string GraphPath, string EnvPath, int initNode, const string& deviceKind, int partition)
{
	loadGraph(GraphPath);
	this->MemSpace = this->graph.vCount;
	this->graph.distance.resize(MemSpace, INT_MAX);
	this->graph.vertexActive.reserve(this->graph.vCount);
	this->graph.vertexActive.assign(this->graph.vCount, 0);
	this->graph.distance[initNode] = 0;
	this->graph.vertexActive[initNode] = 1;
	this->graph.activeNodeNum = 1;

	if (deviceKind == "CPU") {
		this->Engine_CPU(partition);
	}
	else if (deviceKind == "GPU") {
		setEnv(EnvPath);
		this->Engine_GPU(partition);
	}
	else {
		this->Engine_FPGA(partition);
	}
}

void Kruskal::MSGGenMerge_GPU(Graph& g, vector<int>& mValue)
{
	if (g.vCount <= 0) return;

	size_t globalSize = g.eCount;
	cl_int iStatus = 0;
	size_t dim = 1;
	int index = -1;

	int kernelID = 0;
	if (env.nameMapKernel.find("GenMerge") == env.nameMapKernel.end())
	{
		env.nameMapKernel["GenMerge"] = env.setKernel("GenMerge");
		kernelID = env.nameMapKernel["GenMerge"];
		vector<cl_mem> tmp(6, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);
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
		kernelID = env.nameMapKernel["GenMerge"];
	}

	cl_event startEvt;
	index = -1;
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeSrc[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeDst[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeWeight[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
	clWaitForEvents(1, &startEvt);

	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not run kernel");

	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][4], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not reading result buffer");
}

void Kruskal::MSGGenMerge_CPU(Graph& g, vector<int>& mValue)
{
	if (g.vCount <= 0) return;
	for (int i = 0; i < g.eCount; ++i) {
		if ((g.edgeSrc[i] < g.vCount) && (g.edgeDst[i] < g.vCount) && (g.distance[g.edgeSrc[i]] != INT_MAX)&&(g.vertexActive[g.edgeSrc[i]] == 1)) {
			mValue[g.edgeDst[i]] = min(mValue[g.edgeDst[i]],g.edgeWeight[i]);
		}
	}
}

void Kruskal::MSGGenMerge_FPGA(Graph& g, vector<int>& mValue)
{

}
