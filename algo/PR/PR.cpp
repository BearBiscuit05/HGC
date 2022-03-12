#include "PR.h"

PR::PR(std::string GraphPath, std::string EnvPath, std::string deviceKind, int partition)
{
	loadGraph(GraphPath);
	this->MemSpace = this->graph.vCount;
	this->graph.distance.resize(MemSpace, -1);
	for (int i = 0; i < this->graph.vCount; ++i)	this->graph.distance[i] = i;

	this->graph.vertexActive.reserve(this->graph.vCount);
	this->graph.vertexActive.assign(this->graph.vCount, 1);
	this->graph.activeNodeNum = this->graph.vCount;

	if (deviceKind == "CPU") {
		this->Engine_CPU(partition);
	}
	else if (deviceKind == "GPU") {
		setEnv(EnvPath);
		this->Engine_GPU(partition);
	}
	else if (deviceKind == "FPGA") {
		this->Engine_FPGA(partition);
	}
}

void PR::MSGGenMerge_CPU(Graph& g, vector<int>& mValue)
{
}

void PR::MSGGenMerge_GPU(Graph& g, vector<int>& mValue)
{
}

void PR::MSGGenMerge_FPGA(Graph& g, vector<int>& mValue)
{
}
