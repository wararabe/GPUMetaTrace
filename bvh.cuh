#ifndef __BVH_CUH__
#define __BVH_CUH__

const int BVH_STACKSIZE = 100;

typedef struct _SBVH_StackElem{
	int nodeOffset;
	float t_min;
}SBVH_StackElem;

typedef struct _SBVH_Stack{
	SBVH_StackElem Stack[BVH_STACKSIZE];
	int StackPtr;
}SBVH_Stack;

typedef struct _SBVHNode{
	float3 min;//pad: 31: (inner or leaf) flag, 30..29: Axis, 28..0: éqÉmÅ[ÉhÇÃêÊì™index
	int flagindex;
	float3 max;//pad: num_primitives
	int nprimitives;
}SBVHNode;

//operation of stack
__device__ void initStackGPU(SBVH_Stack &io_Stack){

	io_Stack.Stack[0].nodeOffset = 0;
	io_Stack.Stack[0].t_min      = 1.0e20f;
	io_Stack.StackPtr            = 1;

}

__device__ void popStackGPU(SBVH_Stack &io_Stack, int* out_node, float* out_tmin){

	io_Stack.StackPtr--;
	*out_node = io_Stack.Stack[io_Stack.StackPtr].nodeOffset;
	*out_tmin = io_Stack.Stack[io_Stack.StackPtr].t_min;

}

__device__ void pushSortStackGPU(SBVH_Stack &io_Stack, const int in_node, const float in_tmin){

	int i;
	for(i=io_Stack.StackPtr-1;io_Stack.Stack[i].t_min<in_tmin;--i){
		io_Stack.Stack[i+1].nodeOffset = io_Stack.Stack[i].nodeOffset;
		io_Stack.Stack[i+1].t_min      = io_Stack.Stack[i].t_min;
	}
	io_Stack.Stack[i+1].nodeOffset = in_node;
	io_Stack.Stack[i+1].t_min      = in_tmin;
	io_Stack.StackPtr++;

}

#endif