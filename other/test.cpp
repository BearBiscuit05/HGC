#include "algo/algo.h"
#include "env/opencl.h"
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
    //BFS bfs = BFS("data/10kV_100kE.txt", "env/kernel/BFS_GPU.cl",0,0,4);

    int num = 100000000;
    const size_t globalSize = num;
    int dim = 1;
    cl_int iStatus = 0;
    vector<int> nums(num, 2);
    Env env = Env();
    env.setEnv("env/kernel/add.cl");
    clock_t start, end;


    env.setKernel("addNum");
    start = clock();
    cl_mem memA = clCreateBuffer(env.context, CL_MEM_READ_WRITE, num * sizeof(int), nullptr, nullptr);
    if (memA == nullptr)
        env.noPtrCheck(nullptr, "set mem error");
    iStatus = clSetKernelArg(env.kernels[0], 0, sizeof(cl_mem), &memA);
    env.errorCheck(iStatus, "set kernel agrs fail!");
    clEnqueueWriteBuffer(env.queue, memA, CL_TRUE, 0, num * sizeof(int), &nums[0], 0, nullptr, nullptr);
    iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[0], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
    env.errorCheck(iStatus, "Can not run kernel");
    iStatus = clEnqueueReadBuffer(env.queue, memA, CL_TRUE, 0, num * sizeof(int), &nums[0], 0, NULL, NULL);
    env.errorCheck(iStatus, "Can not reading result buffer");
    end = clock();

    for (int i = 0; i < num; ++i) {
        if (nums[i] != 1) {
            cout << "fail count" << endl;
            exit(0);
        }
    }
    cout << "GPU Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;

    start = clock();
    for (int i = 0; i < num; ++i) {
        nums[i] += 1;
    }
    end = clock();
    cout << "CPU Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;


}