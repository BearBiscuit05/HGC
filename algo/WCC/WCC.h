#pragma once
#include <vector>
#include "../../graph/graph.h"
#include "../../env/opencl.h"

class WCC
{
public:
	WCC(string GraphPath, string EnvPath);
	void setGPUEnv(string filePath);
	void loadGraph(string filePath);

	void Engine(int partition);
	void MergeGraph(vector<Graph>& subGraph);
	void MSGGenMerge(Graph& g, vector<int>& mValue);
	void MSGApply(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum(vector<int> activeNodes);

	int MemSpace = 0;
	Graph graph;
	Env env;
};

