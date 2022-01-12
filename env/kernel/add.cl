__kernel void addNum(__global int4 * A)
{
	size_t index = get_global_id(0);
	A[index].x += 1;
	A[index].y += 1;
	A[index].z += 1;
	A[index].w += 1;
}