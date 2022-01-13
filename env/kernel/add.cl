__kernel void Gather1(__global int4* data, __global int* output,
    __local int4* partial_sums) {

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    partial_sums[lid] = data[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = group_size / 2; i > 0; i >>= 1) {
        if (lid < i) {
            partial_sums[lid] += partial_sums[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = partial_sums[0].x + partial_sums[0].y + partial_sums[0].z + partial_sums[0].w;
    }
}

__kernel void Gather2(__global int* input, __global int* output, __local int* cache)
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