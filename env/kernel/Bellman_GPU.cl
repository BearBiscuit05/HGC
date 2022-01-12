__kernel void GenMerge(__global int* srcs, __global int* dsts, __global int* weights,
    __global int* active, __global int* mValue, __global int* vValue,
    __global int* initOfNode, __global int* numOfinit)
{
    size_t index = get_global_id(0);
    if (active[srcs[index]] == 1) {
        for (int i = 0; i < numOfinit[0]; ++i) {
            int init2src = numOfinit[0] * srcs[index] + i;
            int init2dst = numOfinit[0] * dsts[index] + i;
            if (vValue[init2src] != INT_MAX) {
                atomic_min(&mValue[init2dst], vValue[init2src] + weights[index]);
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void Apply(__global int* active, __global int* mValues, __global int* vValues)
{
    size_t dst = get_global_id(1);
    size_t initV = get_global_id(0);
    size_t vCount = get_global_size(1);
    size_t initOfVNum = get_global_size(0);

    int index = dst * initOfVNum + initV;

    if (mValues[index] < vValues[index]) {
        vValues[index] = mValues[index];
        active[dst] = 1;
    }
}


//use for computing active node quantity
__kernel void Gather(__global int* input, __global int* output, __local int* cache)
{
    int lid = get_local_id(0);
    int bid = get_group_id(0);
    int gid = get_global_id(0);
    int localSize = get_local_size(0);

    cache[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize >> 1; i > 0; i >>= 1)
    {
        if (lid < i)
        {
            cache[lid] += cache[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[bid] = cache[0];
    }
}

