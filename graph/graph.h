#pragma once
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
class Vertex
{
public:
    Vertex(int vertexID, bool activeness, int initVIndex);

    int vertexID;
    bool isActive;
};

class Edge
{
public:
    Edge(int src, int dst, double weight);

    int src;
    int dst;
    double weight;
};

class Graph
{
public:
    void insertEdge(int src, int dst, double weight);
    void readFile2Graph(string fileName);
    vector<Graph> divideGraphByEdge(int partition);
    void readFile2NodeGraph(string fileName,int nodeNumber);

    int vCount;
    int eCount;
    int activeNodeNum = 0;

    std::vector<int> edgeSrc;
    std::vector<int> edgeDst;
    std::vector<int> edgeWeight;
    std::vector<int> vertexID;
    std::vector<int> vertexActive;
    std::vector<int> distance;

    vector<Graph> subGraph = vector<Graph>();
};
