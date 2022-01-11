__kernel void addNum(__global int * input , __global int * output)
{
	size_t index = get_global_id(0);
	A[index] = A[index] + 1;
}