#include "bellman.h"

Bellman::Bellman(string GraphPath, string EnvPath, int initNode, int deviceKind, int partition)
{
	loadGraph(GraphPath);
	this->MemSpace = this->graph.vCount;
	this->graph.distance.resize(MemSpace, INT_MAX);
	graph.vertexActive[initNode] = 1;
	graph.distance[initNode] = 0;
	this->graph.activeNodeNum = 1;

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
		vector<cl_mem> tmp(6, nullptr);
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//src
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//dst
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//weight
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.vCount * sizeof(int), nullptr, nullptr);//vertex
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//mValue
		tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//vValue

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
	clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
	clWaitForEvents(1, &startEvt);

	iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not run kernel");

	iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][4], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, NULL, NULL);
	env.errorCheck(iStatus, "Can not reading result buffer");
	
}

void Bellman::Engine_CPU(int partition)
{
	int iter = 0;
	vector<int> mValues(this->MemSpace);
	clock_t start, end, subStart=0, subiter;
	start = clock();
	while (this->graph.activeNodeNum > 0) {
		cout << "----------------------" << endl;
		cout << "this is iter : " << iter++ << endl;
		//subStart = clock();
		//subiter = clock();
		vector<Graph> subGraph = graph.divideGraphByEdge(partition);
		//cout << "divide run time: " << (double)(clock() - subStart) << "ms" << endl;
		for (auto& g : subGraph) {
			mValues.assign(this->MemSpace, INT_MAX);
			//subStart = clock();
			MSGGenMerge_CPU(g, mValues);
			//cout << "Gen run time: " << (double)(clock() - subStart) << "ms" << endl;
			//subStart = clock();
			MSGApply_CPU(g, mValues);
			//cout << "Apply run time: " << (double)(clock() - subStart) << "ms" << endl;
		}
		//subStart = clock();
		MergeGraph(subGraph);
		//cout << "mergeGraph run time: " << (double)(clock() - subStart) << "ms" << endl;
		//subStart = clock();
		this->graph.activeNodeNum = GatherActiveNodeNum_CPU(this->graph.vertexActive);
		//cout << "Gather run time: " << (double)(clock() - subStart) << "ms" << endl;
		//cout << "------------------------------" << endl;
		//cout << "iter run  time: " << (double)(clock() - subiter) << "ms" << endl;
		cout << "active node number" << this->graph.activeNodeNum << endl;
		cout << "------------------------------" << endl;
		if (iter == 21)
			break;
	}
	end = clock();
	cout << "Run time: " << (double)(end - start) << "ms" << endl;
	
}

void Bellman::MSGGenMerge_CPU(Graph& g, vector<int>& mValue)
{
	if (g.vCount == 0)	return;
	for (int i = 0; i < g.eCount; i++)
	{
		if ((g.edgeSrc[i] < g.vCount) && (g.edgeDst[i] < g.vCount) && g.vertexActive[g.edgeSrc[i]] == 1)
		{
			mValue[g.edgeDst[i]] = min(mValue[g.edgeDst[i]],g.distance[g.edgeSrc[i]] + g.edgeWeight[i]);
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
			g.vertexActive[i] = 1;
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
