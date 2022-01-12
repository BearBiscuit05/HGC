#pragma once
#include <vector>
#include "../../graph/graph.h"
#include "../../env/opencl.h"

class WCC
{
public:
	WCC(string GraphPath, string EnvPath, int deviceKind, int partition);
	void setEnv(string filePath);
	void loadGraph(string filePath);
	void MergeGraph(vector<Graph>& subGraph);

	void Engine_CPU(int partition);
	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue);
	void MSGApply_CPU(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_CPU(vector<int>& activeNodes);

	void Engine_GPU(int partition);
	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue);
	void MSGApply_GPU(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_GPU(vector<int>& activeNodes);

	void Engine_FPGA(int partition);
	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);
	void MSGApply_FPGA(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_FPGA(vector<int>& activeNodes);

	int MemSpace = 0;
	Graph graph;
	Env env;
};

