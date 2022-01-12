#include "bellman.h"

Bellman::Bellman(string GraphPath, string EnvPath, vector<int>& vertexNode, int deviceKind, int partition)
{
	loadGraph(GraphPath);
	this->numOfInit = vertexNode.size();
	this->initV = vertexNode;
	this->MemSpace = this->graph.vCount * this->numOfInit;

	this->graph.distance.resize(MemSpace, INT_MAX);

	for (int i = 0; i < this->numOfInit; ++i) {
		graph.vertexActive[initV[i]] = 1;
		graph.distance[initV[i] * numOfInit + i] = 0;
	}
	this->graph.activeNodeNum = numOfInit;

	if (deviceKind == 0) {
		this->Engine_CPU(partition);
	}
	else if (deviceKind == 1) {
		setEnv(EnvPath);
		this->Engine_GPU(partition);
	}
	else {
		this->Engine_FPGA(partition);
	}
}

void Bellman::setEnv(string filePath)
{
	this->env.setEnv(filePath);
	cout << "load GPU env success" << endl;
}

void Bellman::loadGraph(string filePath)
{
	this->graph.readFile2Graph(filePath);
	cout << "load graph success" << endl;
}

void Bellman::MergeGraph(vector<Graph>& subGraph)
{
	this->graph.vertexActive.assign(this->graph.vCount, 0);
	this->graph.activeNodeNum = 0;
	for (auto& g : subGraph) {
		for (int i = 0; i < this->graph.vCount; ++i) {
			this->graph.vertexActive[i] |= g.vertexActive[i];
		}

		for (int i = 0; i < this->MemSpace; ++i) {
			if (this->graph.distance[i] > g.distance[i]) {
				this->graph.distance[i] = g.distance[i];
			}
		}
		this->graph.activeNodeNum += g.activeNodeNum;
	}

}

void Bellman::Engine_GPU(int partition)
{
	int iter = 0;
	vector<int> mValues(this->MemSpace);
	clock_t start, end,substart,subend;
	start = clock();
	while (this->graph.activeNodeNum > 0) {
		cout << "----------------------" << endl;
		cout << "this is GPU iter : " << ++iter << endl;
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		for (auto& g : subGraph) {
			mValues.assign(this->MemSpace, INT_MAX);
			MSGGenMerge_GPU(g, mValues);
			MSGApply_GPU(g,mValues);
		}
		MergeGraph(subGraph);
		this->graph.activeNodeNum = GatherActiveNodeNum_GPU(this->graph.vertexActive);
		cout << "active node number :" << this->graph.activeNodeNum << endl;
	}
	end = clock();
	cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;
}

void Bellman::MSGGenMerge_GPU(Graph &g,vector<int> &mValue)
{
	if (g.vCount <= 0) return ;
	
	size_t globalSize = g.eCount;
	cl_int iStatus = 0;
	size_t dim = 1;
	int index = -1;

	int kernelID = 0;
	if (env.kernels.size() == 0) {
		kernelID = env.setKernel("GenMerge");
		vector<cl_mem> tmp(8, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//src
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//dst
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//weight
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.vCount * sizeof(int), nullptr, nullptr);//vertex
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//mValue
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//vValue
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->numOfInit * sizeof(int), nullptr, nullptr);//initV
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, nullptr);//numofInitV

		for (int i = 0; i <= index; i++) {
			if (tmp[i] == nullptr)
				env.noPtrCheck(nullptr, "set mem error");
		}
		env.clMem.push_back(tmp);

		for (int i = 0; i <= index; i++) {
			iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
		}
		env.errorCheck(iStatus, "set kernel agrs fail!");
	} else {
		kernelID = 0;
	}
	
	cl_event startEvt;
	index = -1;
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeSrc[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeDst[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeWeight[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0],  0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->numOfInit * sizeof(int), &this->initV[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, sizeof(int), &this->numOfInit, 0, nullptr, &startEvt);
	clWaitForEvents(1, &startEvt);

	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not run kernel");

	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][4], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not reading result buffer");
	
}

void Bellman::MSGApply_GPU(Graph& g, vector<int>& mValue)
{
	fill(g.vertexActive.begin(), g.vertexActive.end(), 0);
	g.activeNodeNum = 0;

	int kernelID = 0 , index = 0;
	const size_t globalSize[2] = { this->numOfInit,g.vCount };
	cl_int iStatus = 0;
	size_t dim = 2;
	if (env.kernels.size() == 1) {
		kernelID = env.setKernel("Apply");
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
	} else {
		kernelID = 1;
	}
	
	cl_event startEvt;
	index = -1;
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
	clWaitForEvents(1, &startEvt);

	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, globalSize, nullptr, 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not run kernel");

	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, NULL, NULL);
	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][2], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not reading result buffer");
}

int Bellman::GatherActiveNodeNum_GPU(vector<int>& activeNodes)
{
	int kernelID = 0, index = 0;
	const size_t localSize = 1024;
	int len = activeNodes.size();
	int group = (len - 1) / localSize + 1;

	const size_t globalSize = group * localSize;
	activeNodes.resize(globalSize, 0);
	cl_int iStatus = 0;
	size_t dim = 1;
	vector<int> subSum(group, 0);
	if (env.kernels.size() == 2) {
		kernelID = env.setKernel("Gather");
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
		kernelID = 2;
	}

	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, globalSize * sizeof(int), &activeNodes[0], 0, nullptr, nullptr);
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, nullptr, nullptr);

	env.errorCheck(clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
		"Can not run kernel2");

	env.errorCheck(clEnqueueReadBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, NULL, NULL),
		"Can not reading result buffer");

	int sum = 0;
	for (int i = 0; i < group; ++i) {
		sum += subSum[i];
	}
	return sum;
}

void Bellman::Engine_CPU(int partition)
{
	int iter = 0;
	vector<int> mValues(this->MemSpace);
	clock_t start, end;
	start = clock();
	while (this->graph.activeNodeNum > 0) {
		cout << "----------------------" << endl;
		cout << "this is CPU iter : " << ++iter << endl;
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		for (auto& g : subGraph) {
			mValues.assign(this->MemSpace, INT_MAX);
			MSGGenMerge_CPU(g, mValues);
			MSGApply_CPU(g, mValues);
		}
		MergeGraph(subGraph);
		cout << "active node number :" << this->graph.activeNodeNum <<endl;
	}
	end = clock();
	cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;
	
}

void Bellman::MSGGenMerge_CPU(Graph& g, vector<int>& mValue)
{
	if (g.vCount == 0)	return;
	for (int i = 0; i < g.eCount; i++)
	{
		if (g.vertexActive[g.edgeSrc[i]] == 1)
		{
			for (int j = 0; j < this->numOfInit; j++)
			{
				int init2src = g.edgeSrc[i] * this->numOfInit + j;
				int init2dst = g.edgeDst[i] * this->numOfInit + j;
				if ((g.distance[init2src] != INT_MAX) && (mValue[init2dst] > g.distance[init2src] + g.edgeWeight[i]))
					mValue[init2dst] = g.distance[init2src] + g.edgeWeight[i];
			}
		}
	}
}

void Bellman::MSGApply_CPU(Graph& g, vector<int>& mValue)
{
	if (g.vCount == 0)	return;
	fill(g.vertexActive.begin(), g.vertexActive.end(), 0);
	g.activeNodeNum = 0;

	for (int i = 0; i < this->MemSpace; i++)
	{
		if (g.distance[i] > mValue[i])
		{
			g.distance[i] = mValue[i];
			g.vertexActive[i / this->numOfInit] = 1;
			g.activeNodeNum++;
		}
	}
}

int Bellman::GatherActiveNodeNum_CPU(vector<int>& vec)
{
	int len = vec.size(), ans = 0;
	for (int i = 0; i < len; ++i) {
		ans += vec[i];
	}
	return ans;
}

void Bellman::Engine_FPGA(int partition)
{
	int iter = 0;
	vector<int> mValues;
	clock_t start, end;
	start = clock();
	while (this->graph.activeNodeNum > 0) {
		cout << "this is iter : " << iter++ << endl;
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		for (auto& g : subGraph) {
			mValues.resize(this->MemSpace, INT_MAX);
			MSGGenMerge_GPU(g, mValues);
			MSGApply_GPU(g, mValues);
		}
		MergeGraph(subGraph);
	}
	end = clock();
	cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;
}

void Bellman::MSGGenMerge_FPGA(Graph& g, vector<int>& mValue)
{

}

void Bellman::MSGApply_FPGA(Graph& g, vector<int>& mValue)
{

}

int Bellman::GatherActiveNodeNum_FPGA(vector<int>& activeNodes)
{
	return 0;
}
