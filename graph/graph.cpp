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
    this->vCount++;
    this->vertexID.resize(vCount, 0);
    this->vertexActive.resize(vCount, 0);
    this->edgeSrc.reserve(this->eCount);
    this->edgeDst.reserve(this->eCount);
    this->edgeWeight.reserve(this->eCount);
    for (int i = 0; i < this->vCount; ++i)   this->vertexID[i] = i;
    for (int i = 0; i < eCount; i++)
    {
        char s;
        int dst, src;
        int weight;
        Gin >>s>>src >> dst >> weight;
        this->insertEdge(src, dst, weight);
    }
    Gin.close();
}

vector<Graph> Graph::divideGraphByEdge(int partition)
{
    Graph g;
    vector<Graph> res(partition,g);

    for (int i = 0; i < partition; ++i) {
        res.at(i).vCount = this->vCount;
        res.at(i).vertexID = this->vertexID;
        res.at(i).distance = this->distance;
        res.at(i).vertexActive = this->vertexActive;
        res.at(i).activeNodeNum = this->activeNodeNum;
        for (int k = i * this->eCount / partition; k < (i + 1) * this->eCount / partition; k++)
            res.at(i).insertEdge(this->edgeSrc.at(k), this->edgeDst.at(k), this->edgeWeight.at(k));
        res.at(i).eCount = res.at(i).edgeSrc.size();
    }
    return res;
}
