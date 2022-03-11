#include "algo/algoHead.h"
#include "env/opencl.h"

using namespace std;
int main() {
    //Bellman bellman = Bellman("data/10kV_100kE.txt", "env/kernel/Bellman_GPU.cl", 0, 0, 2);
    //BFS bfs = BFS("data/10kV_100kE.txt", "env/kernel/BFS_GPU.cl", 0,0,2);
    //Kruskal kruskal = Kruskal("data/10kV_100kE.txt", "env/kernel/Kruskal_GPU.cl", 0,0,2);
    WCC wcc = WCC("../data/10kV_100kE.txt", "../env/kernel/WCC_GPU.cl",0,2);

}