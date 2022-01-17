#pragma once
#include <vector>
#include "../algo.h"

class Kruskal : public Algo
{
public:
	Kruskal(string GraphPath, string EnvPath,int initNode, int deviceKind, int partition);
	void MergeGraph(vector<Graph>& subGraph);

	void Engine_CPU(int partition);
	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue);
	void MSGApply_CPU(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_CPU(vector<int>& activeNodes);

	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue);
	void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue) {};

	void Engine_FPGA(int partition);
	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);
	void MSGApply_FPGA(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_FPGA(vector<int>& activeNodes);
};

