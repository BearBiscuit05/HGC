#include "graph.h"


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
        for (int i = 0; i < partition; ++i) {
            subGraph.at(i).vCount = this->vCount;
            subGraph.at(i).vertexID = this->vertexID;
            subGraph.at(i).distance = this->distance;
            subGraph.at(i).vertexActive = this->vertexActive;
            subGraph.at(i).activeNodeNum = this->activeNodeNum;
            for (int k = i * this->eCount / partition; k < (i + 1) * this->eCount / partition; k++)
                subGraph.at(i).insertEdge(this->edgeSrc.at(k), this->edgeDst.at(k), this->edgeWeight.at(k));
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
