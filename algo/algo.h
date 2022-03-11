#pragma once
#include <vector>
#include "../graph/graph.h"
#include "../env/opencl.h"


class Algo
{
public:
    Algo(string GraphPath, string EnvPath, const string& deviceKind, int partition);
	void setEnv(string filePath);
	void loadGraph(string filePath);

	void Engine_GPU(int partition);
	virtual void MSGGenMerge_GPU(Graph& g, vector<int>& mValue) {}
	virtual void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue) {}
	void MergeGraph_GPU(vector<Graph>& subGraph);
	void MSGApply_GPU(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_GPU(vector<int>& activeNodes);

	void Engine_CPU(int partition);
	void MergeGraph_CPU(vector<Graph>& subGraph);
	void MSGApply_CPU(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum_CPU(vector<int>& activeNodes);
	virtual void MSGGenMerge_CPU(Graph& g, vector<int>& mValue) {}

    virtual void Engine_FPGA(int partition);
    virtual void MergeGraph_CPU(vector<Graph>& subGraph);
    virtual void MSGApply_CPU(Graph& g, vector<int>& mValue);
    virtual int GatherActiveNodeNum_CPU(vector<int>& activeNodes);
    virtual void MSGGenMerge_CPU(Graph& g, vector<int>& mValue) {}
    int MemSpace = 0;
	Graph graph;
	Env env;
};





