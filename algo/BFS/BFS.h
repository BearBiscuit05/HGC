#pragma once
#include <vector>
#include "../algo.h"

using namespace std;
class BFS : public Algo
{
public:
	BFS();
	BFS(string GraphPath, string EnvPath, int initNode, int deviceKind, int partition);

	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue);

	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue);
	void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue);


	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);


};

