#pragma once
#include <vector>
#include "../algo.h"

using namespace std;
class WCC : public Algo
{
public:
	WCC(string GraphPath, string EnvPath, string deviceKind, int partition) :
        Algo(GraphPath,EnvPath,deviceKind,partition) {};

	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue) override;

	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue) override;
	void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue) override {};

	void Engine_FPGA(int partition);
	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);
	void MSGApply_FPGA(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_FPGA(vector<int>& activeNodes);
};

