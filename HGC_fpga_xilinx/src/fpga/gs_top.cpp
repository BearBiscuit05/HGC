#include "fpga_mem.h"

extern "C" {
	void gs_top(
			int* edgeSrcArray,
			int* vertexValue,
			int* edgeDstArray,
			int* tmpvertexValue,
			int edgeNum,
			int vertexBegin,
			int vertexNum4pe,
			int activeNum
			)
	{
#pragma HLS INTERFACE m_axi port=edgeSrcArray offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=edgeSrcArray bundle=control

#pragma HLS INTERFACE m_axi port=edgeDstArray offset=slave bundle=gmem2 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=edgeDstArray bundle=control

#pragma HLS INTERFACE m_axi port=tmpvertexValue offset=slave bundle=gmem1 max_read_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE s_axilite port=tmpvertexValue bundle=control

#pragma HLS INTERFACE m_axi port=vertexValue offset=slave bundle=gmem1 max_read_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE s_axilite port=vertexValue bundle=control

#pragma HLS INTERFACE s_axilite port=vertexBegin       bundle=control
#pragma HLS INTERFACE s_axilite port=edgeNum       bundle=control
#pragma HLS INTERFACE s_axilite port=vertexNum4pe       bundle=control
#pragma HLS INTERFACE s_axilite port=activeNum       bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

	int srcsList[MAX_EDGE_IN_ONE_PARTITION];
	int dstList[MAX_EDGE_IN_ONE_PARTITION];
	int oldVertexValuePool[MAX_VERTEX_IN_ONE_PARTITION];
	int newVertexValuePool[MAX_VERTEX_IN_ONE_PARTITION];
	int activeNodeNum = 0;
	int eCount = edgeNum;
	int vCount = vertexNum4pe;
	int firstId = vertexBegin;

	for(int i = 0 ; i < MAX_EDGE_IN_ONE_PARTITION;i++)
	{
		srcsList[i] = edgeSrcArray[i];
	}

	for(int i = 0 ; i < MAX_EDGE_IN_ONE_PARTITION ;i++)
	{
		dstList[i] = edgeDstArray[i];
	}

	for(int i = 0 ; i < MAX_VERTEX_IN_ONE_PARTITION;i++)
	{
		oldVertexValuePool[i] = vertexValue[i];
	}
	
	for(int i = 0 ; i < MAX_VERTEX_IN_ONE_PARTITION ; i++)
	{
		newVertexValuePool[i] = oldVertexValuePool[i];
	}

	for(int i = 0  ; i < MAX_EDGE_IN_ONE_PARTITION ; i++)
	{
#pragma HLS pipeline
		int src = srcsList[i];
		int dst = dstList[i];
		int oldV = newVertexValuePool[dst];
		int newV = oldVertexValuePool[src];
		if(newV > oldV){
			newVertexValuePool[dst] = newV;
			activeNodeNum++;
		}
	}

	activeNum = activeNodeNum;
	for(int i = 0 ; i < MAX_VERTEX_IN_ONE_PARTITION;i++)
	{
		tmpvertexValue[i] = newVertexValuePool[i];
	}
	tmpvertexValue[MAX_VERTEX_IN_ONE_PARTITION] = activeNodeNum;
 	}
}
