#ifndef _FPGA_MEM_H_
#define	_FPGA_MEM_H_
#include "config.h"


template <typename T>
void burstRead2Stream(
		int16* DDRdata,
		hls::stream<T>& input,
		int& edgeNum
)
{
#pragma HLS function_instantiate variable=DDRdata
#pragma HLS function_instantiate variable=input
	int16 readBuffer[READBUFFERSIZE];
	int loopCount = ((edgeNum - 1) >> (LOG_BANDWIDTH_INT_SIZE + LOG_READBUFFER_SIZE)) + 1;  
read_loop:
	for(int i = 0 ; i < loopCount ; i++)
	{
read_to_GM_loop:
		for(int j = 0 ; j < READBUFFERSIZE ; j++)
		{
			readBuffer[j] = DDRdata[(i << LOG_READBUFFER_SIZE )+ j];
		}
 read_to_stream_loop:
 		for(int k = 0 ; k < READBUFFERSIZE ; k++)
 		{
 			input << readBuffer[k];
		}
	}
}

template <typename T>
void SliceEdgeStream(
	hls::stream<T>& input,
	hls::stream<T>& output,
	int& edgeNum
)
{
#pragma HLS function_instantiate variable=input
	int loopNum = edgeNum;
fifo_loop:
	for(int i = 0 ; i < loopNum ; i++)
	{
#pragma HLS PIPELINE II=1
		T tmp;
		tmp = input.read();
		output << tmp;
	}
}

template <typename T>
void SliceVDStream(
	hls::stream<T>& input,
	hls::stream<T>& output
)
{
#pragma HLS function_instantiate variable=input
	T V;
	while(true)
	{
		V = input.read();
		output << V;
		if(V.dstId == -1){
			break;
		}
	}
}


template <typename T>
inline int clear_stream (hls::stream<T> &stream)
{
#pragma HLS INLINE
    int end_counter = 0;
clear_stream: while (true)
    {
        T clear_data;

        if ( read_from_stream_nb(stream, clear_data) == 0)
        {
            end_counter ++;
        }
        if (end_counter > 256)
        {
            break;
        }
    }
    return 0;
}

template <typename T>
inline int read_from_stream_nb (hls::stream<T> &stream, T & value)
{
#pragma HLS INLINE
    if (stream.empty())
    {
        return 0;
    }
    else
    {
        value = stream.read();
        return 1;
    }
}

template <typename T>
void StreamDuplicate4(
	hls::stream<T> &input,
	hls::stream<T> &output1,
	hls::stream<T> &output2,
	hls::stream<T> &output3,
	hls::stream<T> &output4,
	int& edgeNum
)
{
	#pragma HLS function_instantiate variable=input
	T V ;
	int loopNum = edgeNum;
	for(int i = 0 ; i < loopNum ; i++)
	{
#pragma HLS PIPELINE II=1
		V = input.read();
		output1 << V;
		output2 << V;
		output3 << V;
		output4 << V;
	}
}


void GetValueFromArray(
	int i,
	int16* array,
	int& index,
	int& ans
)
{
#pragma HLS function_instantiate variable=i
	int col_index = (index >> 4);
	int row_index = (index & 15); 
	ans = array[col_index].range(row_index*32+31,row_index*32);
}

void makeVDstTuple(
	hls::stream<int16> &srcStream,
	hls::stream<int16> &dstStream,
	hls::stream<int> &vertexValue,
	hls::stream<v_dst_tuples_t> &v_dst_tuples,
	int& edgeNum
) 
{
	int loopNum = edgeNum >> 4;
	int16 srcList,dstList;
	int srcIndex = -1 , srcValue = 0;
	int srcArray[INT_SIZE_IN_ONE_BANDWIDTH] , dstArray[INT_SIZE_IN_ONE_BANDWIDTH];
	for(int i = 0; i < loopNum ; i++)
	{
		srcList = srcStream.read();
		dstList = dstStream.read();
		for(int j = 0 ; j < INT_SIZE_IN_ONE_BANDWIDTH ; j++){
#pragma HLS UNROLL
			srcArray[j] = srcList.range(j*32+31,j*32);	
		}
		for(int j = 0 ; j < INT_SIZE_IN_ONE_BANDWIDTH ; j++){
#pragma HLS UNROLL
			dstArray[j] = dstList.range(j*32+31,j*32);	
		}
		for(int k = 0 ; k < INT_SIZE_IN_ONE_BANDWIDTH ; k++){
			while(srcIndex != srcArray[k]){
				srcValue = vertexValue.read();
				srcIndex++;
			}
			v_dst_tuples_t v_dst_tuple;
			v_dst_tuple.value = srcValue;
			v_dst_tuple.dstId = dstArray[k];
			v_dst_tuples << v_dst_tuple;
		}
	}

	int T;
	while(!vertexValue.empty())
	{
		T = vertexValue.read();
	}
}

void transform16To1(
	int16* values,
	int* value
)
{	
	int V = 0;
	for(int i = 0  ; i < 8 ; i++)
	{
		for(int j = 0 ; j < 16 ; j++)
		{
			V = values[i].range(j*32+31,j*32);
			value[i *16 + j] = V ;
		}
	}
}

void transform1To16(
	int16* values,
	int* value,
	int& eCount
)
{
	int16 tmpV;
	int V = 0;
	for(int i = 0  ; i < (eCount >> 4) ; i++)
	{
		for(int j = 0 ;  j < 16 ; j++)
		{
			V = value[i*16 + j];
			tmpV.range(j*32 + 31 , j*32) = V;
		}
		values[i] = tmpV;
	}
}

void burstRead2BufferPool(
	int16* vertexValue,
	int* vertexValuePool,
	int& vertexNum
)
{
	int loop = vertexNum >> 4;
	int16 tmp;
	for(int i = 0 ; i < loop ; i++)
	{
		tmp = vertexValue[i];
		for(int j = 0 ; j < PE_NUM ; j++)
		{
#pragma HLS UNROLL
			vertexValuePool[i*16 + j] = tmp.range(j*32+31,j*32);
		}
	}
}

void buffer2Stream(
	int* vertexValuePool,
	hls::stream<int> &vertexValueStream ,
	int& vertexNum
)
{
	int V = 0;
	for(int i = 0 ; i < vertexNum ; i++)
	{
#pragma HLS PIPELINE II=1		
		V = vertexValuePool[i];
		vertexValueStream << V;
	}	
}

void tupleShuffle(
	int &index,
	hls::stream<v_dst_tuples_t> &v_dst_tuples_input,
	hls::stream<v_dst_tuples_t> &v_dst_tuples_output,
	int &allEdgeNum
)
{
#pragma HLS function_instantiate variable=index
	int exitNum = index;
	v_dst_tuples_t tmp;
	int dstID = 0 , V = 0;
	for(int i = 0 ; i < allEdgeNum ; i++)
	{
		tmp = v_dst_tuples_input.read();
		dstID = tmp.dstId;
		if((dstID&15) == exitNum)
		{
			v_dst_tuples_output << tmp;
		}
	}
	tmp.dstId = -1;
	v_dst_tuples_output << tmp;
}

void vertexValueMerge(
	int* oldVertexValuePool,
	int (*VertexValuePool)[MAX_VERTEX_IN_ONE_PE],
	int &vertexBegin,
	int &vertexNum,
	int &activeNodeNum
)
{
	int loopNum = vertexNum >> 4;
	int arrayList[16];
	for(int i = 0 ; i < loopNum ; i++)
	{
		for(int j = 0 ; j < PE_NUM ; j++)
		{
#pragma HLS UNROLL
			arrayList[j] = VertexValuePool[j][i];
		}

		for(int k = 0 ; k < PE_NUM ; k++)
		{
#pragma HLS UNROLL
			int oldIndex = vertexBegin + i*16 + k;
			int oldV = oldVertexValuePool[oldIndex];
			if(arrayList[k] > oldV)
			{
				//std::cout << "index :" << oldIndex <<" ,oldV :" <<oldV << ", k:"<<k<< "\n";
				oldVertexValuePool[oldIndex] = arrayList[k];
				activeNodeNum++;
			}
		}
	}
}

void compute(
    int &index,
    int &vertexBegin,
    int &vertexNumInPE,
    hls::stream<v_dst_tuples_t>& VDtuples,
    int* valueCache
)
{
#pragma HLS function_instantiate variable=index
    v_dst_tuples_t vdT;
    int arrayIndex = 0, lastV = 0;
    for(int i = 0 ; i < MAX_VERTEX_IN_ONE_PE ; i++)
	{
		valueCache[i] = 0;
	}

	while(true)
    {
        vdT = VDtuples.read();
		if(vdT.dstId == -1)
            break;
        arrayIndex = (vdT.dstId - vertexBegin) >> 4;
        if(vdT.value > lastV)
			valueCache[arrayIndex] = vdT.value; // core
		// std::cout <<"CU index :"<< index << " now VD tuples [value:" 
		// << vdT.value << ",dst:"<<vdT.dstId<<"]"<< " , and it store in :" << arrayIndex <<",and value is : "
		// <<valueCache[arrayIndex] <<std::endl; 
    }
}

#endif
