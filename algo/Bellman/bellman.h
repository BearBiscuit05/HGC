#pragma once
#include <vector>
#include "../../graph/graph.h"
#include "../../env/opencl.h"
using namespace std;

class Bellman
{
public:
	Bellman(string GraphPath,string EnvPath,vector<int> &vertexNode);
	void setGPUEnv(string filePath);
	void loadGraph(string filePath);
	void Engine(int partition);
	void MergeGraph(vector<Graph> &subGraph);
	void MSGGenMerge(Graph& g, vector<int>& mValue);
	void MSGApply(Graph& g, vector<int>& mValue);
	int GatherActiveNodeNum(vector<int> activeNodes);


	vector<int> initV;
	int numOfInit = 0;
	int MemSpace = 0;
	Graph graph;
	Env env;
};

