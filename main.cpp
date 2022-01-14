#include "env/opencl.h"
#include "algo/Bellman/bellman.h"
#include "algo/WCC/WCC.h"
#include "algo/BFS/BFS.h"
#include "algo/Kruskal/Kruskal.h"
//#include <iostream>
//#include <vector>

using namespace std;
int main() {
    {
        /*vector<int> initV = { 0,2,4 };
        Bellman bellman = Bellman("data/USA-road-d.BAY.gr", "env/kernel/Bellman_GPU.cl", initV,0,10);*/

        //WCC wcc = WCC("data/USA-road-d.BAY.gr", "env/kernel/WCC_GPU.cl");
        //wcc.Engine(20);

        //BFS bfs = BFS("data/testGraph.txt", "env/kernel/BFS_GPU.cl",100);
        //bfs.Engine(10);
        //Kruskal kruskal = Kruskal("data/testGraph.txt", "env/kernel/Kruskal_GPU.cl");
        //kruskal.Engine(2);

        //Env env = Env();
        //env.showDeviceInfo();
    }
    Bellman bellman = Bellman("data/roadNet-PA.txt", "env/kernel/Bellman_GPU.cl", 0,1,2);
}