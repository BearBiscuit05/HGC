#include "algo/algo.h"
#include "env/opencl.h"
//#include <iostream>
//#include <vector>

using namespace std;
int main() {
    clock_t start, end;

    int num = 1048576 * 64;
    int dim = 1;
    cl_int iStatus = 0;
    vector<int> nums(num, 2);
    const size_t localSize = 1024;
    int group = num / localSize;
    const size_t globalSize = num / 4;
    vector<int> sum(group / 4, 0);

    Env env = Env();
    env.setEnv("env/kernel/add.cl");
    env.setKernel("Gather1");

    cl_mem memA = clCreateBuffer(env.context, CL_MEM_READ_WRITE, num * sizeof(int), nullptr, nullptr);
    cl_mem memB = clCreateBuffer(env.context, CL_MEM_READ_WRITE, group * sizeof(int), nullptr, nullptr);
    if (memA == nullptr || memB == nullptr)
        env.noPtrCheck(nullptr, "set mem error");

    iStatus = clSetKernelArg(env.kernels[0], 0, sizeof(cl_mem), &memA);
    iStatus |= clSetKernelArg(env.kernels[0], 1, sizeof(cl_mem), &memB);
    iStatus |= clSetKernelArg(env.kernels[0], 2, localSize * 4 * sizeof(int), nullptr);
    env.errorCheck(iStatus, "set kernel agrs fail!");
    start = clock();
    clEnqueueWriteBuffer(env.queue, memA, CL_TRUE, 0, num * sizeof(int), &nums[0], 0, nullptr, nullptr);
    clEnqueueWriteBuffer(env.queue, memB, CL_TRUE, 0, group * sizeof(int), &sum[0], 0, nullptr, nullptr);

    iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[0], 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    env.errorCheck(iStatus, "Can not run kernel");
    iStatus = clEnqueueReadBuffer(env.queue, memB, CL_TRUE, 0, group / 4 * sizeof(int), &sum[0], 0, NULL, NULL);
    env.errorCheck(iStatus, "Can not reading result buffer");
    end = clock();

    int ans1 = 0;
    for (int i = 0; i < group / 4; ++i) {
        ans1 += sum[i];
    }
    cout << "GPU Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;

    start = clock();
    int ans = 0;
    for (int i = 0; i < num; ++i) {
        ans += nums[i];
    }
    end = clock();
    cout << "CPU Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << endl;

    if (ans1 == ans)
        cout << "count success" << endl;
    else
        cout << ans1 << "!= " << ans << endl;
    return 0;
}