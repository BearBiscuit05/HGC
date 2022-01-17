#include "algo/algoHead.h"
#include "env/opencl.h"
//#include <iostream>
//#include <vector>

using namespace std;
int main() {
    {
        /*Bellman bellman = Bellman("data/USA-road-d.BAY.gr", "env/kernel/Bellman_GPU.cl", initV,0,10);*/

        //WCC wcc = WCC("data/USA-road-d.BAY.gr", "env/kernel/WCC_GPU.cl");
        //wcc.Engine(20);

        //BFS bfs = BFS("data/testGraph.txt", "env/kernel/BFS_GPU.cl",100);
        //bfs.Engine(10);
        //Kruskal kruskal = Kruskal("data/testGraph.txt", "env/kernel/Kruskal_GPU.cl");
        //kruskal.Engine(2);

        //Env env = Env();
        //env.showDeviceInfo();
    }
    //Bellman bellman = Bellman("data/10kV_100kE.txt", "env/kernel/Bellman_GPU.cl",0, 1, 2);
    BFS bfs = BFS("data/Amazon0601.txt", "env/kernel/BFS_GPU.cl",0,1,2);
    int ans = 0;
    for (int i = 0; i < bfs.graph.vCount; ++i) {
        if (bfs.graph.distance[i] == 0)  ans++;
    }
    cout << "ans: "<<ans << endl;
    //Kruskal kruskal = Kruskal("data/testGraph.txt", "env/kernel/Kruskal_GPU.cl",0,1,2);
    //WCC wcc = WCC("data/10kV_100kE.txt", "env/kernel/WCC_GPU.cl",1,2);
}