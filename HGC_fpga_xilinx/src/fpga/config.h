#ifndef _CONFIG_H_
#define	_CONFIG_H_

#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
typedef ap_int<512> int16;
typedef ap_int<64> uram_bw;

#define INT_MASK 2147483647
#define READBUFFERSIZE 128
#define LOG_BANDWIDTH_INT_SIZE 4
#define LOG_READBUFFER_SIZE 7 
#define PE_NUM 16
#define INT_SIZE_IN_ONE_BANDWIDTH 16
#define LOG_PE 4 
//#define MAX_VERTEX_IN_ONE_PARTITION 524288
#define MAX_VERTEX_IN_ONE_PARTITION 32
#define MAX_EDGE_IN_ONE_PARTITION (MAX_VERTEX_IN_ONE_PARTITION*2)
#define MAX_VERTEX_IN_ONE_PE (((MAX_VERTEX_IN_ONE_PARTITION - 1)>>(LOG_PE)) + 1)

typedef struct
{
    int value;
    int dstId;
} v_dst_tuples_t;

typedef struct
{
    int src;
    int dst;
}src_dst_tuples_t;
#endif
