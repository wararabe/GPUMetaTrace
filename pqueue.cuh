#ifndef __PQUEUE_CUH__
#define __PQUEUE_CUH__

#include"hitrecord.cuh"

//operation of priorityQueue
__device__ void priorityQueuePush(const SHitRecord *hits, const int n_hits, int* s_ids, int &s_n){

	int i=s_n;

	s_ids[s_n] = n_hits;
	s_n        = s_n+1;

	while(i!=0){
		int j=(i-1)/2;
		if(hits[s_ids[i]].ray_t<hits[s_ids[j]].ray_t){
			int tmp =s_ids[i];
			s_ids[i]=s_ids[j];
			s_ids[j]=tmp;
			i=j;
		} else break;
	}

}

__device__ int priorityQueuePop(const SHitRecord *hits, int *s_ids, int &s_n){

	int i=0,j;
	int id = s_ids[0];
	s_n = s_n-1;
	s_ids[0]= s_ids[s_n];

	while(s_n>(2*i+1)){
		j=2*i+1;
		if((j!=s_n-1) && hits[s_ids[j]].ray_t>hits[s_ids[j+1]].ray_t)++j;
		if(hits[s_ids[i]].ray_t>hits[s_ids[j]].ray_t){
			int tmp=s_ids[i];
			s_ids[i]=s_ids[j];
			s_ids[j]=tmp;
			i=j;
		} else break;
	}

	return id;

}

__device__ inline float priorityQueueNextT(const SHitRecord *hits, const int *s_ids){
	return hits[s_ids[0]].ray_t;
}

#endif