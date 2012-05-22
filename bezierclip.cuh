#ifndef __BEZIERCLIP_CUH__
#define __BEZIERCLIP_CUH__

#include"hitrecord.cuh"

//#define DENSITY_GPU    0.3f
#define DENSITY_GPU    0.9f
//#define DENSITY_GPU    0.5f

typedef struct _SBCStack{

	float data[5][7];
	float tmin[5];
	float tmax[5];
	int ptr;

}SBCStack;

//operation of bezierclip
__device__ void subdivide(float v[], const int n, const float t){

	for(int i=0;i<n-1;++i)
		for(int j=n-1;j>i;--j)
			v[j] += (1.0f-t)*(v[j-1]-v[j]);

}

__device__ void subdivide2(float v[], const int n, const float t){

	for(int i=0;i<n-1;++i)
		for(int j=0;j<n-1-i;++j)
			v[j] += t*(v[j+1]-v[j]);

}

__device__ void calc_ctrlpGPU(const SSphAccel *spheres, const float3 &o, const float3 &d, const float back, float* ctrlp, const int* ids, const SHitRecord* hits, const int n_ids, const float front){

	for(int i=0;i<7;++i) ctrlp[i] = -DENSITY_GPU;

	for(int i=0;i<n_ids;++i){

		const SHitRecord hit = hits[ids[i]];

		const float w   = 2.0f*hit.rD;
		const float d1  = back-hit.min_t;

		if(back>hit.min_t+w+1.0e-4f) continue;

		const int x     = hit.primitive_id;

		const float a   = (hit.rD/spheres[x].r)*(hit.rD/spheres[x].r);
		const float a_2 = a*a;
		float tmp[7]    = {0.0f,0.0f,16.0f/27.0f*a_2,8.0f/45.0f*(8.0f*a+5.0f)*a_2,16.0f/27.0f*a_2,0.0f,0.0f};

		const float front_t = d1/w;
		if(front_t>1.0e-6f)
			subdivide2(tmp,7,front_t);
		const float back_t  = (front-back)/(w-d1);
		if(back_t>1.0e-6f)
			subdivide(tmp,7,back_t);

		for(int j=0;j<7;++j) ctrlp[j] += tmp[j];

	}

}

__device__ float getPointGPU(const int n, const float p[], const float t){

	float tmp[7];
	for(int i=0;i<6;++i)
		tmp[i] = t*p[i+1]+(1.0f-t)*p[i];

	for(int i=1;i<n-1;++i)
		for(int j=0;j<n-1-i;++j)
			tmp[j] += t*(tmp[j+1]-tmp[j]);

	return tmp[0];
}

__device__ float secantGPU(const float tmin, const float tmax, const float p[], const int n){

	const int max_loop_counter = 15;
	float lt,rt,ly,ry;
	float t;

	if(p[0]<0.0f){
		ly=p[0];   lt=0.0f;
		ry=p[n-1]; rt=1.0f;
	} else {
		ly=p[n-1]; lt=1.0f;
		ry=p[0];   rt=0.0f;
	}

	for(int i_loop=0;i_loop<max_loop_counter;++i_loop){

		t = rt - ry * (rt-lt)/(ry-ly);
		float tmp = getPointGPU(n,p,t);

		if(fabsf(tmp)<1.0e-6f) return (tmax-tmin)*t+tmin;

		if(tmp<0.0f){
			ly=tmp;
			lt=t;
		} else {
			ry=tmp;
			rt=t;
		}

	}

	return (tmax-tmin)*t+tmin;

}

__device__ void pushBCStackGPU(SBCStack *bcstack, const float *ctrlp, const float tmin, const float tmax, const float rate){

	memcpy(bcstack->data[bcstack->ptr],ctrlp,sizeof(float)*7);
	subdivide2(bcstack->data[bcstack->ptr],7,0.5f);
	bcstack->tmin[bcstack->ptr] = tmin;
	bcstack->tmax[bcstack->ptr] = tmax;
	++(bcstack->ptr);

}

__device__ void popBCStackGPU(SBCStack *bcstack, float *ctrlp, float *tmin, float *tmax){

	--(bcstack->ptr);
	memcpy(ctrlp,bcstack->data[bcstack->ptr],sizeof(float)*7);
	*tmin = bcstack->tmin[bcstack->ptr];
	*tmax = bcstack->tmax[bcstack->ptr];

}

__device__ float raySurfIntersectGPU(float *ctrlp, const int n_ctrlp, float tmin, float tmax){

	SBCStack bcstack;
	bcstack.ptr=0;

	while(1){

		int n_sign_changes=0;
		int i;
		for(i=0;i<n_ctrlp-1;++i){
			if(ctrlp[i]*ctrlp[i+1]<0.0f)
				++n_sign_changes;
		}

		if(n_sign_changes==0){

			if(bcstack.ptr<=0) return -1.0f;
			popBCStackGPU(&bcstack,ctrlp,&tmin,&tmax);

		} else if(n_sign_changes>1){
			const float rate = 0.5f;//(float)(i+1)/(float)n_ctrlp;
			const float tmid = tmin + (tmax-tmin)*rate;
			pushBCStackGPU(&bcstack,ctrlp,tmid,tmax,1.0f-rate);
			subdivide(ctrlp,n_ctrlp,rate);
			tmax = tmid;

		} else {

			float t = secantGPU(tmin,tmax,ctrlp,n_ctrlp);
			return t;

		}

	}

}

#endif