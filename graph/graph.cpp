#include "graph.h"
#include <unordered_map>

Vertex::Vertex(int vertexID, bool activeness, int initVIndex)
{
    this->vertexID = vertexID;
    this->isActive = activeness;
}

Edge::Edge(int src, int dst, double weight)
{
    this->src = src;
    this->dst = dst;
    this->weight = weight;
}

void Graph::insertEdge(int src, int dst, double weight)
{
    this->edgeSrc.emplace_back(src);
    this->edgeDst.emplace_back(dst);
    this->edgeWeight.emplace_back(weight);
}

void Graph::readFile2Graph(string fileName) {
    std::ifstream Gin(fileName);
    if (!Gin.is_open()) { std::cout << "Error! File not found!" << std::endl; exit(0); }
    Gin >> this->vCount >> this->eCount;
    this->vCount = ((this->vCount - 1) / 1024 + 1) *1024;
    this->vertexID.resize(vCount, 0);
    this->vertexActive.resize(vCount, 0);
    this->edgeSrc.reserve(this->eCount);
    this->edgeDst.reserve(this->eCount);
    this->edgeWeight.reserve(this->eCount);
    for (int i = 0; i < this->vCount; ++i)   this->vertexID[i] = i;
    for (int i = 0; i < eCount; i++)
    {
        int dst, src;
        Gin >>src >> dst ;
        this->insertEdge(src, dst, 1);
    }
    Gin.close();
}

vector<Graph> Graph::divideGraphByEdge(int partition)
{
    if (subGraph.size() == 0) {
        Graph g;
        subGraph.resize(partition, g);
        int iter = this->edgeDst.size();
        int part = this->edgeSrc.size() / iter;
        for (int i = 0; i < partition; ++i) {
            subGraph.at(i).vCount = this->vCount;
            subGraph.at(i).vertexID = this->vertexID;
            subGraph.at(i).distance = this->distance;
            subGraph.at(i).vertexActive = this->vertexActive;
            subGraph.at(i).activeNodeNum = this->activeNodeNum;
            if (part == 1) {
                for (int k = i * this->eCount / partition; k < (i + 1) * this->eCount / partition; k++)
                    subGraph.at(i).insertEdge(this->edgeSrc.at(k), this->edgeDst.at(k), this->edgeWeight.at(k));
            }
            else {    
                for (int k = i * iter / partition; k < (i + 1) * iter / partition; k++) {
                    subGraph.at(i).edgeDst.push_back(this->edgeDst[k]);
                    for (int j = 0; j < part; ++j) {
                        subGraph.at(i).edgeSrc.push_back(4*k + j);
                        subGraph.at(i).edgeWeight.push_back(1);
                    }
                }
            }
            subGraph.at(i).eCount = subGraph.at(i).edgeSrc.size();
        }
    }
    else {
        for (int i = 0; i < partition; ++i) {
            subGraph.at(i).distance = this->distance;
            subGraph.at(i).vertexActive = this->vertexActive;
            subGraph.at(i).activeNodeNum = this->activeNodeNum;
        }
        
    }
    return this->subGraph;
}

void Graph::readFile2NodeGraph(string fileName, int nodeNumber)
{
    std::unordered_map<int, std::vector<int>> GatherEdge;
    std::ifstream Gin(fileName);
    if (!Gin.is_open()) { std::cout << "Error! File not found!" << std::endl; exit(0); }
    Gin >> this->vCount >> this->eCount;
    this->vCount = ((this->vCount - 1) / 1024 + 1) * 1024;
    this->vertexID.resize(vCount, 0);
    this->vertexActive.resize(vCount, 0);
    this->edgeSrc.reserve(this->eCount);
    this->edgeDst.reserve(this->eCount);
    this->edgeWeight.reserve(this->eCount);
    for (int i = 0; i < this->vCount; ++i)   this->vertexID[i] = i;
    
    for (int i = 0; i < eCount; i++)
    {
        int dst, src;
        Gin >> src >> dst;
        GatherEdge[dst].push_back(src);
        if (GatherEdge[dst].size() == nodeNumber) {
            this->edgeSrc.insert(this->edgeSrc.end(), GatherEdge[dst].begin(), GatherEdge[dst].end());
            this->edgeDst.push_back(dst);
            GatherEdge[dst].clear();
        }
    }
    Gin.close();

    for (auto& edge : GatherEdge) {
        this->edgeDst.push_back(edge.first);
        edge.second.resize(nodeNumber, edge.first);
        this->edgeSrc.insert(this->edgeSrc.end(), edge.second.begin(), edge.second.end());
    }
    this->eCount = this->edgeSrc.size();
    this->edgeWeight.resize(this->eCount, 1);   
}

