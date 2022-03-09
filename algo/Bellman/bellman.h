#pragma once
#include <vector>
#include "../algo.h"
using namespace std;

class Bellman : public Algo
{
public:
	Bellman(string GraphPath,string EnvPath,int initNode,int deviceKind,int partition);
	void MergeGraph(vector<Graph>& subGraph);

	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue);

	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue);
	void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue) {};

	void Engine_FPGA(int partition);
	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);
	void MSGApply_FPGA(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_FPGA(vector<int>& activeNodes);

};

