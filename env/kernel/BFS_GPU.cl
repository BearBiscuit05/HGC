__kernel void GenMerge(__global int* srcs, __global int* dsts,__global int* active, __global int* mValue)
{
    size_t index = get_global_id(0);
    if(active[srcs[index]] == 1) {
        atomic_min(&mValue[dsts[index]], 0);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void GenMergeByNode(__global int* srcs, __global int* dsts, __global int* active, __global int* mValues,__global int* args)
{
    size_t index = get_global_id(0);
    size_t globalSize = get_global_size(0);
    int dstId = dsts[index];
    int m = INT_MAX;

    for (int i = 0; i < 4; ++i) {
        int srcID = srcs[index * 4 + i];
        if (srcID == dstId)
            break;
        if (active[srcID] == 1) {
            m = 0;
            break;
        }
    }
    atomic_min(&mValues[dsts[index]], m);
}

__kernel void Apply(__global int* active, __global int* mValues, __global int* vValues)
{
    size_t index = get_global_id(0);
    if (mValues[index] < vValues[index]) {
        vValues[index] = mValues[index];
        active[index] = 1;
    }
    else {
        active[index] = 0;
    }
}

__kernel void MergeGraph(__global int* active_1, __global int* active_2, __global int* distance_1, __global int* distance_2)
{
    size_t index = get_global_id(0);
    active_1[index] |= active_2[index];
    distance_1[index] = min(distance_1[index],distance_2[index]);
}

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