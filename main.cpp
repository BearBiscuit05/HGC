#include "algo/algoHead.h"
#include "env/opencl.h"
//#include <iostream>
//#include <vector>

using namespace std;
int main() {
    //Bellman bellman = Bellman("data/soc-pokec-relationships.txt", "env/kernel/Bellman_GPU.cl", 46, 1, 2);
    //BFS bfs = BFS("data/Amazon0601.txt", "env/kernel/BFS_GPU.cl", 0,1,2);
    //Kruskal kruskal = Kruskal("data/soc-pokec-relationships.txt", "env/kernel/Kruskal_GPU.cl", 46,1,2);
    WCC wcc = WCC("data/Amazon0601.txt", "env/kernel/WCC_GPU.cl",0,2);
}