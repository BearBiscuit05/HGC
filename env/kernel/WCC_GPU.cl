__kernel void GenMerge(__global int* srcs, __global int* dsts, __global int* weights,
    __global int* active, __global int* mValue, __global int* vValue)
{
    size_t index = get_global_id(0);
    if (active[srcs[index]] == 1) {
        atomic_min(&mValue[dsts[index]], vValue[srcs[index]]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void Apply(__global int* active, __global int* mValues, __global int* vValues)
{
    size_t index = get_global_id(0);
    if (mValues[index] < vValues[index]) {
        vValues[index] = mValues[index];
        active[index] = 1;
    }
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

