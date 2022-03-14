#include <vector>
#include "../algo.h"

class PR : public Algo
{
public:
	PR(std::string GraphPath, std::string EnvPath, std::string deviceKind, int partition);

	void MSGGenMerge_CPU(Graph& g, vector<int>& mValue);
	void MSGGenMerge_GPU(Graph& g, vector<int>& mValue);
	void MSGGenMergeByNode_GPU(Graph& g, vector<int>& mValue) {};
	void MSGGenMerge_FPGA(Graph& g, vector<int>& mValue);
};
