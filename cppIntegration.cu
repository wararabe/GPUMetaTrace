#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<cmath>
#include<iostream>
#include<fstream>
#include<sstream>
#include<GL/glut.h>
#include<GL/glui.h>
#include<cutil.h>
#include<vector_types.h>

typedef float3 SColorRGB;

typedef struct _SRay{
	float3 origin;
	float d_min;
	float3 direction;
	float d_max;
}SRay;

typedef struct _SSphAccel{
	float3 o;
	float r_2;
	int primitive_id;
	float r;
	int move_flag;
	float speed;
	SColorRGB c;
}SSphAccel;

#include "cppIntegration_kernel.cu"
#include "SVector3.cuh"
#include "CQuaternion.cuh"
#include "Camera.cuh"
#include "bvh.cuh"
#include "pqueue.cuh"
#include "bezierclip.cuh"

//device variables
//__device__ __constant__ int ref_type = 0;
#define MAX_NUM_HITS   500
#define MAX_NUM_IDS    100

//#define KERNEL_RATE    0.396765f
#define KERNEL_RATE    0.0422765f
//#define KERNEL_RATE    0.5f
#define WINDOWX        640
#define WINDOWY        480

__device__ SSphAccel *spheres_GPU;
__device__ SBVHNode  *BVH_Nodes;

__device__ SColorRGB *colors2;
__device__ float *background;

/*
__device__ inline void kernelRayBoxIntersect(const ray_t& in_Ray, const float3& in_Min, const float3& in_Max, float& t_near, float& t_far){

	float l1 = __fdividef(in_Min.x - in_Ray.pos.x, in_Ray.dir.x);
	float l2 = __fdividef(in_Max.x - in_Ray.pos.x, in_Ray.dir.x);
	t_near = fmaxf(fminf(l1,l2), t_near);
	t_far = fminf(fmaxf(l1,l2), t_far);
	l1 = __fdividef(in_Min.y - in_Ray.pos.y, in_Ray.dir.y);
	l2 = __fdividef(in_Max.y - in_Ray.pos.y, in_Ray.dir.y);
	t_near = fmaxf(fminf(l1,l2), t_near);
	t_far = fminf(fmaxf(l1,l2), t_far);
	l1 = __fdividef(in_Min.z - in_Ray.pos.z, in_Ray.dir.z);
	l2 = __fdividef(in_Max.z - in_Ray.pos.z, in_Ray.dir.z);
	t_near = fmaxf(fminf(l1,l2), t_near);
	t_far = fminf(fmaxf(l1,l2), t_far);

}

*/

__device__ void RayBoxIntersectGPU(const SRay *in_ray, const SVector3 *in_min, const SVector3 *in_max, float *io_min, float *io_max){

	float t[2][3];
	t[0][0] = __fdividef(in_min->x-in_ray->origin.x, in_ray->direction.x);
	t[0][1] = __fdividef(in_min->y-in_ray->origin.y, in_ray->direction.y);
	t[0][2] = __fdividef(in_min->z-in_ray->origin.z, in_ray->direction.z);
	t[1][0] = __fdividef(in_max->x-in_ray->origin.x, in_ray->direction.x);
	t[1][1] = __fdividef(in_max->y-in_ray->origin.y, in_ray->direction.y);
	t[1][2] = __fdividef(in_max->z-in_ray->origin.z, in_ray->direction.z);

	const int sel0 = (in_ray->direction.x < 0.0f);
	const int sel1 = (in_ray->direction.y < 0.0f);
	const int sel2 = (in_ray->direction.z < 0.0f);

	*io_max = min(min(t[1-sel2][2], *io_max), min(t[1-sel1][1], t[1-sel0][0]));
	*io_min = max(max(t[sel2][2], *io_min), max(t[sel1][1], t[sel0][0]));

}

__device__ void raySphereIntersectGPU(const float t, const SSphAccel *s, const SRay *in_Ray, SHitRecord *hit, const int id){

	const float oc_x = in_Ray->origin.x - s[id].o.x;
	const float oc_y = in_Ray->origin.y - s[id].o.y;
	const float oc_z = in_Ray->origin.z - s[id].o.z;

	const float d_oc = in_Ray->direction.x*oc_x + in_Ray->direction.y*oc_y + in_Ray->direction.z*oc_z;
	const float oc_oc = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z;

	const float D = d_oc * d_oc - oc_oc + s[id].r_2;

	if(D<0.0f) return;

	const float sqrtf_D = sqrtf(D);
	hit->min_t = -sqrtf_D - d_oc;

	if(-sqrtf_D-d_oc>t) {
		hit->ray_t = -sqrtf_D - d_oc;
		hit->primitive_id = id;
	} else if(sqrtf_D-d_oc>t){
		hit->ray_t =  sqrtf_D - d_oc;
		hit->primitive_id = -id;
	} else {
		return;
	}

	hit->rD = sqrtf_D;

}

__device__ void raySphereIntersectRecalcGPU(const SSphAccel *s, const SRay *in_Ray, SHitRecord *hit, const int id){

	const float oc_x = in_Ray->origin.x - s[id].o.x;
	const float oc_y = in_Ray->origin.y - s[id].o.y;
	const float oc_z = in_Ray->origin.z - s[id].o.z;

	const float d_oc = in_Ray->direction.x*oc_x + in_Ray->direction.y*oc_y + in_Ray->direction.z*oc_z;
	const float oc_oc = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z;

	const float D = d_oc * d_oc - oc_oc + s[id].r_2;

	const float sqrtf_D = sqrtf(D);
	hit->min_t        = -sqrtf_D - d_oc;
	hit->ray_t        = -sqrtf_D - d_oc;
	hit->primitive_id = id;
	hit->rD           = sqrtf_D;

}

__device__ inline float rayKernelSphereIntersectGPU(const float t, const SSphAccel *s, const SRay *in_Ray, const int id){

	const float oc_x  = in_Ray->origin.x - s[id].o.x;
	const float oc_y  = in_Ray->origin.y - s[id].o.y;
	const float oc_z  = in_Ray->origin.z - s[id].o.z;

	const float d_oc  = in_Ray->direction.x*oc_x + in_Ray->direction.y*oc_y + in_Ray->direction.z*oc_z;
	const float oc_oc = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z;

	const float D = d_oc * d_oc - oc_oc + s[id].r_2 * KERNEL_RATE;

	const float sqrtf_D = sqrtf(D);

	if(D<0.0f) return 1.0e20f;

	if(-sqrtf_D-d_oc>t) {
		return -sqrtf_D - d_oc;
	} else if(sqrtf_D-d_oc>t){
		return  sqrtf_D - d_oc;
	} else {
		return 1.0e20f;
	}

}

//operation of raytrace
__device__ void calc_normalGPU(SVector3 &normal, const SSphAccel *s, const float3 &o, const int *ids, const SHitRecord *hits, const int n_ids){

	normal = setVector(0.0f);

	for(int i=0;i<n_ids;++i){
		const int   p_id    = hits[ids[i]].primitive_id;
		const float inv_r2  = 1.0f / (s[p_id].r*s[p_id].r);
		const float dir[3]  = {s[p_id].o.x-o.x,
			s[p_id].o.y-o.y,
			s[p_id].o.z-o.z};
		const float r2  = (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2])*inv_r2;
		const float e = ((-24.0f/9.0f*r2+68.0f/9.0f)*r2-44.0f/9.0f)*inv_r2;
		normal.x += e * dir[0];
		normal.y += e * dir[1];
		normal.z += e * dir[2];
	}

	normal = unitVector(normal);

}

__device__ void calc_refract(SRay *ray, const SVector3 &normal){

	const float water_n   = 2.14f;

	float RN = dot(ray->direction,normal);

	if(RN<0.0f){
		float D = 1.0f-(1.0f-RN*RN)/(water_n*water_n);
		//air to water
		ray->direction = (-RN/water_n-sqrtf(D))*normal + 1.0f/water_n * ray->direction;

	} else {
		float D = 1.0f-(1.0f-RN*RN)*(water_n*water_n);
		//water to air
		if(D<0.0f){
			ray->direction  = ray->direction + 2.0f * RN * normal;
		} else {
			ray->direction = (-RN*water_n+sqrtf(D))*normal + water_n*ray->direction;
		}
	}

}

__device__ void calc_reflect(SRay *ray, const SVector3 &normal){

	float RN = ray->direction.x*normal.x + ray->direction.y*normal.y + ray->direction.z*normal.z;

	ray->direction.x = ray->direction.x - 2.0f * RN * normal.x;
	ray->direction.y = ray->direction.y - 2.0f * RN * normal.y;
	ray->direction.z = ray->direction.z - 2.0f * RN * normal.z;

}

__device__ inline float rayBVHIntersectGPU(const SSphAccel *spheres, const SBVHNode *m_Nodes, SRay *in_Ray, int *ids, int &n_ids, SHitRecord *hits, int *sorted_ids, int &n_sorted_ids, int &chit){

	SBVH_Stack io_Stack;
	float t_near     = in_Ray->d_min;
	float t_far      = in_Ray->d_max;

	RayBoxIntersectGPU(in_Ray, &(m_Nodes[0].min), &(m_Nodes[0].max), &t_near, &t_far);

	if(t_near>=t_far) return 0.0f;

	initStackGPU(io_Stack);
	const SBVHNode* the_Node = &m_Nodes[0];

	const int flg[3]   = {in_Ray->direction.x>=0.0f, in_Ray->direction.y>=0.0f, in_Ray->direction.z>=0.0f};
	const int back[3]  = {flg[0],flg[1],flg[2]};
	const int front[3] = {1-back[0],1-back[1],1-back[2]};

	float tmin            = t_near;
	float tmax            = in_Ray->d_max;
	const float start_min = t_near;

	while(1){

		while((the_Node->flagindex & 0x80000000) == 0){

			float t_min_near = start_min; float t_min_far = start_min;
			float t_max_near = tmax;      float t_max_far = tmax;

			const int axis = (the_Node->flagindex & 0x60000000) >> 29;

			const int _near_id = (the_Node->flagindex&0x1fffffff) + front[axis];
			const int _far_id  = (the_Node->flagindex&0x1fffffff) + back[axis];

			RayBoxIntersectGPU(in_Ray, &(m_Nodes[_near_id].min), &(m_Nodes[_near_id].max), &t_min_near, &t_max_near);
			RayBoxIntersectGPU(in_Ray, &(m_Nodes[_far_id].min),  &(m_Nodes[_far_id].max),  &t_min_far,  &t_max_far);

			const int intersect_near = (t_min_near< t_max_near);
			const int intersect_far  = (t_min_far < t_max_far);

			if(!intersect_near && !intersect_far) {
				break;
			} else if(!intersect_near) {
				the_Node = &(m_Nodes[_far_id]);
			} else 	if(!intersect_far) {
				the_Node = &(m_Nodes[_near_id]);
			} else {
				pushSortStackGPU(io_Stack, _far_id, t_min_far);
				the_Node = &(m_Nodes[_near_id]);
			}

		}

		if((the_Node->flagindex & 0x80000000) != 0){

			for(int i=the_Node->flagindex & 0x7fffffff; i<(the_Node->flagindex & 0x7fffffff)+the_Node->nprimitives; i++){

				tmax = max(tmax, rayKernelSphereIntersectGPU(start_min, spheres, in_Ray, i));
				hits[chit].ray_t = 1.0e20f;
				raySphereIntersectGPU(start_min, spheres, in_Ray, &hits[chit], i);

				if(hits[chit].ray_t < tmax){

					priorityQueuePush(hits,chit,sorted_ids,n_sorted_ids);
					chit = chit % (MAX_NUM_HITS-1) + 1;

					/*
					if(hits[chit].primitive_id>0){
					priorityQueuePush(hits,chit,sorted_ids,n_sorted_ids);
					memcpy(&hits[chit+1],&hits[chit],sizeof(SHitRecord));
					hits[chit+1].primitive_id =  -i;
					hits[chit+1].ray_t        += 2.0f*hits[chit+1].rD;
					chit = chit % MAX_NUM_HITS + 1;
					}

					priorityQueuePush(hits,chit,sorted_ids,n_sorted_ids);
					chit = chit % MAX_NUM_HITS + 1;
					*/
				}

			}
		}

		int the_NodePtr;
		popStackGPU(io_Stack, &the_NodePtr, &tmin);
		the_Node = &m_Nodes[the_NodePtr];

		while(priorityQueueNextT(hits,sorted_ids)<tmin){

			int id = priorityQueuePop(hits,sorted_ids,n_sorted_ids);

			SHitRecord hit = hits[id];

			float td_t=-1.0f;
			float isect_t;

			if(n_ids>1){

				float ctrlp[7];
				calc_ctrlpGPU(spheres,in_Ray->origin,in_Ray->direction,in_Ray->d_min,ctrlp,ids,hits,n_ids,hit.ray_t);
				td_t    = raySurfIntersectGPU(ctrlp,7,0.0f,1.0f);
				isect_t = (hit.ray_t-in_Ray->d_min)*td_t + in_Ray->d_min;

			} else if(n_ids==1){

				isect_t = rayKernelSphereIntersectGPU(in_Ray->d_min, spheres, in_Ray, hits[ids[0]].primitive_id);
				td_t    = isect_t<hit.ray_t?1.0f:0.0f;

			}

			if(td_t>0.0f) return isect_t;

			in_Ray->d_min = hit.ray_t;

			if(hit.primitive_id>0){
				ids[n_ids++] = id;
				memcpy(&hits[chit],&hit,sizeof(SHitRecord));
				hits[chit].primitive_id = -hit.primitive_id;
				hits[chit].ray_t        += 2.0f*hit.rD;
				priorityQueuePush(hits,chit,sorted_ids,n_sorted_ids);
				chit = chit % (MAX_NUM_HITS-1) + 1;

			} else {
				int i;
				for(i=0;i<n_ids;++i)
					if(hits[ids[i]].primitive_id==-hit.primitive_id)
						break;
				ids[i] = ids[--n_ids];
			}

		}
		//		}

		if(io_Stack.StackPtr < 1) return 0.0f;

	}

}

__global__ void cameraInit(){


	camera.m_EyePoint    = setVector(0.000f,1.000f,-2.5f);
	//	camera.m_EyePoint    = setVector(0.000f,0.000f,-13.0f);
	camera.m_FocalLength = 0.034f;
	camera.m_IrisF       = 0.3f;

	const SVector3 in_v = setVector(1.000f,0.500f,0.000f);
	//	const SVector3 in_v = setVector(0.000f,0.000f,0.000f);
	const SVector3 v    = in_v-camera.m_EyePoint;
	const float len = length(v);

	camera.m_FocusLength     = len;
	camera.m_HorizontalAngle = atan2f(-v.z,v.x);
	camera.m_VerticalAngle   = asinf(v.y/len);
	camera.m_RoundAngle      = 0.0f;

	camera.camera_x = setVector(0.0f,0.0f,1.0f);
	camera.camera_y = setVector(0.0f,1.0f,0.0f);
	camera.camera_z = setVector(-1.0f,0.0f,0.0f);

	//rotate with y-axis
	camera.camera_x = rotateVector(camera.camera_x, camera.camera_y, camera.m_HorizontalAngle);
	camera.camera_z = rotateVector(camera.camera_z, camera.camera_y, camera.m_HorizontalAngle);

	//rotate with x-axis
	camera.camera_y = rotateVector(camera.camera_y, camera.camera_x, camera.m_VerticalAngle);
	camera.camera_z = rotateVector(camera.camera_z, camera.camera_x, camera.m_VerticalAngle);

	//rotate with z-axis
	camera.camera_x = rotateVector(camera.camera_x, camera.camera_z, -camera.m_RoundAngle);
	camera.camera_y = rotateVector(camera.camera_y, camera.camera_z, -camera.m_RoundAngle);

	camera.m_LensRadius = camera.m_FocalLength/(float(2.0) * camera.m_IrisF);
	camera.m_T_i = camera.m_FocalLength*camera.m_FocusLength/(camera.m_FocusLength-camera.m_FocalLength);
	camera.m_T_Focus_p_i = camera.m_FocusLength/camera.m_T_i;
	camera.m_Width  = WINDOWX;
	camera.m_Height = WINDOWY;

	float screen_width;
	float screen_height;

	if(camera.m_Width >= float(1.5)*camera.m_Height){
		screen_width = float(0.036);
		screen_height = camera.m_Height * float(0.036) / camera.m_Width;
	} else {
		screen_width = camera.m_Width * float(0.024) / camera.m_Height;
		screen_height = float(0.024);
	}

	camera.p1 = camera.m_EyePoint + camera.m_T_i  * camera.camera_z
		- float(0.5) * screen_width  * camera.camera_x
		+ float(0.5) * screen_height * camera.camera_y;
	camera.p2 = camera.m_EyePoint + camera.m_T_i  * camera.camera_z
		+ float(0.5) * screen_width  * camera.camera_x
		+ float(0.5) * screen_height * camera.camera_y;
	camera.p3 = camera.m_EyePoint + camera.m_T_i  * camera.camera_z
		- float(0.5) * screen_width  * camera.camera_x
		- float(0.5) * screen_height * camera.camera_y;
	camera.p4 = camera.m_EyePoint + camera.m_T_i  * camera.camera_z
		+ float(0.5) * screen_width  * camera.camera_x
		- float(0.5) * screen_height * camera.camera_y;

	camera.m_T_InvWidth  = float(1.0) / float(camera.m_Width);
	camera.m_T_InvHeight = float(1.0) / float(camera.m_Height);

	camera.m_Jitter   = screen_width / camera.m_Width;
	const float inv_jitter  = float(1.0) / camera.m_Jitter;

	camera.m_T_Lr_i   = camera.m_LensRadius * camera.m_T_i * inv_jitter;
	camera.m_T_Lr_ips = camera.m_T_Lr_i / camera.m_FocusLength;

}

__device__ void getColorGPU(const float *bg, const SRay &ray, SColorRGB *c){

	const int back_w = 1000;
	const int back_h = 1000;

	const float x = ray.direction.x;
	const float y = ray.direction.y;
	const float z = ray.direction.z;

	const float l = sqrtf(x*x+y*y+z*z);
	const float dx = x/l;
	const float dy = -y/l;
	const float dz = z/l;

	const float m = 2.0f * sqrtf(dx*dx+dy*dy+(dz+1.0f)*(dz+1.0f));

	const float u = dx / m + 0.5f;
	const float v = dy / m + 0.5f;

	const float w = (back_w-1) * u;
	const float h = (back_h-1) * v;

	const int wi = floorf(w);
	const int hi = floorf(h);

	const float red   = bg[4*(wi+back_w*hi)];
	const float green = bg[4*(wi+back_w*hi)+1];
	const float blue  = bg[4*(wi+back_w*hi)+2];

	c->x = red;
	c->y = green;
	c->z = blue;

}

__global__ void rayTraceGPU(const SSphAccel *spheres, const SBVHNode *m_Nodes, SColorRGB *colors, const float* bg){

	const int x  = blockIdx.x*16 + threadIdx.x;
	const int y  = blockIdx.y*16 + threadIdx.y;
	const int p  = x + y * WINDOWX;

	SRay ray;
	cameraScreenView(x,y,&ray);
	int n_ids;
	int ids[50];
	int sorted_ids[500];
	int n_sorted_ids;
	int chit;
	SHitRecord hits[500];
	SHitRecord hits2[100];

	ray.d_min     = 0.0f;
	ray.d_max     = 1.0e20f;
	n_ids         = 0;
	hits[0].ray_t = 1.0e20f;
	sorted_ids[0] = 0;
	n_sorted_ids  = 1;
	chit          = 1;

	colors[p] = setVector(0.0f,0.0f,0.0f);

	for(int i=0;i<1;++i){

		float isect_t = rayBVHIntersectGPU(spheres, m_Nodes, &ray, ids, n_ids, hits, sorted_ids, n_sorted_ids, chit);

		if(!(isect_t>0.0f)){

			getColorGPU(bg,ray,&colors[p]);
			return;

		} else {

			ray.origin.x += ray.direction.x*isect_t;
			ray.origin.y += ray.direction.y*isect_t;
			ray.origin.z += ray.direction.z*isect_t;

			SVector3 normal;
			calc_normalGPU(normal,spheres,ray.origin,ids,hits,n_ids);

			float L = ray.direction.x * normal.x + ray.direction.y * normal.y + ray.direction.z * normal.z;

			const int ref_type = 1;
			if(ref_type){
				calc_refract(&ray,normal);
			} else { 
				calc_reflect(&ray,normal);
			}
			/*
			in_Ray->direction.x = normal.x;
			in_Ray->direction.y = normal.y;
			in_Ray->direction.z = normal.z;
			*/
			ray.d_min = 1.0e-4f;
			//*
			SColorRGB c = setVector(0.0f);
			/*
			for(int q=0;q<n_ids;++q){

			const SSphAccel s = spheres[hits[ids[q]].primitive_id];
			const float rate = sqrtf((ray.origin.x-s.o[0])*(ray.origin.x-s.o[0])
			+ (ray.origin.y-s.o[1])*(ray.origin.y-s.o[1])
			+ (ray.origin.z-s.o[2])*(ray.origin.z-s.o[2])) / s.r;

			c += (1.0f-rate)*setVector(s.c.x,s.c.y,s.c.z);

			}
			c = fabsf(L) * unitVector(c);

			colors[p] = c;
			*/

			getColorGPU(bg,ray,&colors[p]);

			//			colors[p] = setVector(fabs(ray.direction.x),fabs(ray.direction.y),fabs(ray.direction.z));

			/*
			const float d = sqrtf(ray.direction.x*ray.direction.x	+ ray.direction.y*ray.direction.y	+ ray.direction.z*ray.direction.z);
			ray.direction.x/=d; ray.direction.y/=d;	ray.direction.z/=d;

			for(int j=0;j<n_ids;++j){
			const int id = ids[j];
			raySphereIntersectRecalcGPU(spheres,&ray,&hits2[j],hits[id].primitive_id);
			ids[j]=j+1;
			}
			memcpy(hits+1,hits2,sizeof(SHitRecord)*50);

			sorted_ids[0] = 0;
			n_sorted_ids  = 1;
			chit          = n_ids+1;
			*/
		}

	}

}

__host__ void initDevice(const int argc, const char **argv){
	CUT_DEVICE_INIT(argc,argv);
}

__host__ void initResource(const SSphAccel *in_s, const int s_size, const SBVHNode *in_nodes, const int n_node, const float *in_background){

	if(spheres_GPU!=NULL)
		CUDA_SAFE_CALL(cudaFree(spheres_GPU));
	CUDA_SAFE_CALL(cudaMalloc((void**) &spheres_GPU, s_size*sizeof(SSphAccel)));
	CUDA_SAFE_CALL(cudaMemcpy(spheres_GPU, in_s, s_size*sizeof(SSphAccel), cudaMemcpyHostToDevice) );

	if(BVH_Nodes!=NULL)
		CUDA_SAFE_CALL(cudaFree(BVH_Nodes));
	CUDA_SAFE_CALL(cudaMalloc((void**) &BVH_Nodes, sizeof(SBVHNode)*n_node));
	CUDA_SAFE_CALL(cudaMemcpy(BVH_Nodes, in_nodes, sizeof(SBVHNode)*n_node, cudaMemcpyHostToDevice) );

	if(background!=NULL)
		CUDA_SAFE_CALL(cudaFree(background));
	CUDA_SAFE_CALL(cudaMalloc((void**) &background, sizeof(float)*1000*1000*4));
	CUDA_SAFE_CALL(cudaMemcpy(background, in_background, sizeof(float)*1000*1000*4, cudaMemcpyHostToDevice) );

	if(colors2!=NULL)
		CUDA_SAFE_CALL(cudaFree(colors2));
	CUDA_SAFE_CALL(cudaMalloc((void**) &colors2, sizeof(SColorRGB)*WINDOWX*WINDOWY));

	cameraInit<<<1,1>>>();
	CUT_CHECK_ERROR("Kernel execution failed");

}

__host__ void delDevice(){

	if(spheres_GPU!=NULL)
		CUDA_SAFE_CALL(cudaFree(spheres_GPU));
	if(BVH_Nodes!=NULL)
		CUDA_SAFE_CALL(cudaFree(BVH_Nodes));
	if(background!=NULL)
		CUDA_SAFE_CALL(cudaFree(background));
	if(colors2!=NULL)
		CUDA_SAFE_CALL(cudaFree(colors2));

}

__host__ void runTest(SColorRGB *out_colors){

	dim3 grid(WINDOWX/16,WINDOWY/16,1);
	//	dim3 grid(20,15,1);
	dim3 threads(16,16,1);

	rayTraceGPU<<< grid, threads >>>(spheres_GPU,BVH_Nodes,colors2,background);
	CUT_CHECK_ERROR("Kernel execution failed");

	CUDA_SAFE_CALL(cudaMemcpy((void*)out_colors, (void*)colors2, sizeof(SColorRGB)*WINDOWX*WINDOWY, cudaMemcpyDeviceToHost) );

}

char filename[1000];
char back_filename[1000];
const int WINDOWX=640;
const int WINDOWY=480;
SColorRGB colors[WINDOWY][WINDOWX];
float *background;
int back_w;
int back_h;
float gamma          = 1.8f;
int TRACE_RAY_NUM    = 1;
int PER_PIXEL_RAY    = 1;
int traverse_type    = 3;
int ref_type         = 1;
int n_spheres        = 10000;
float init_r         = 0.015f;
float DENSITY        = 0.9f;
int root_find_method = 0;
int isSAH            = 1;
int kindOfBVH        = 0;
int kindOfObject     = 0;
int isGPU            = 0;
bool is_in           = true;

CBVH bvh;
CBoundingVolume scene_bv;
__declspec(align(16)) SSphAccel  *spheres;

int main(int argc, char *argv[]){

	if(argc==2){
		n_spheres = atoi(argv[1]);
	}

	const CAnsiString& back_filename(BACKGROUND_FILENAME);
	CHdrFileIO hdrio;
	hdrio.read(back_filename,&back_w,&back_h,&background);
	init(0);
	initDevice(argc,(const char**)argv);

	glutInit(&argc,argv);
	glutInitWindowSize(WINDOWX,WINDOWY);
	main_window = glutCreateWindow("(x,y) coordinate");
	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);

	GLUI_Master.set_glutIdleFunc(idle);

	GLUI *glui = GLUI_Master.create_glui("control",0,WINDOWX+10,0);
	GLUI_Spinner *numofsp_spinner = glui->add_spinner("num of sphere",GLUI_SPINNER_INT,&n_spheres,0,init);
	numofsp_spinner->set_int_limits(1,1000000,GLUI_LIMIT_WRAP);
	GLUI_Spinner *numoftr_spinner = glui->add_spinner("Num of trace",GLUI_SPINNER_INT,&TRACE_RAY_NUM);
	numoftr_spinner->set_int_limits(1,20,GLUI_LIMIT_WRAP);
	GLUI_Spinner *numofden_spinner = glui->add_spinner("Threshold",GLUI_SPINNER_FLOAT,&DENSITY);
	numofden_spinner->set_float_limits(0,1.0f,GLUI_LIMIT_WRAP);

	GLUI_Rotation *view_rot  = glui->add_rotation("EnvMap Rotation",envmap_rotate_mat);//,'r',keyboard);
	GLUI_Rotation *view_rot2 = glui->add_rotation("RayDir Rotation",raydir_rotate_mat);//,'r',keyboard);

	glui->add_column(true);
	GLUI_Panel *obj_panel = glui->add_panel("Traverse type");
	GLUI_RadioGroup *group1 = glui->add_radiogroup_to_panel(obj_panel,&traverse_type,traverse_type,control_cb);
	glui->add_radiobutton_to_group(group1,"None");
	glui->add_radiobutton_to_group(group1,"Stack copy");
	glui->add_radiobutton_to_group(group1,"No BVH");
	glui->add_radiobutton_to_group(group1,"Sort");

	GLUI_Panel *ref_panel = glui->add_panel("Reflect or Refract");
	GLUI_RadioGroup *group2 = glui->add_radiogroup_to_panel(ref_panel,&ref_type,ref_type,control_cb);
	glui->add_radiobutton_to_group(group2,"relect");
	glui->add_radiobutton_to_group(group2,"refract");

	GLUI_Panel *rootfind_panel = glui->add_panel("Root find method");
	GLUI_RadioGroup *group3 = glui->add_radiogroup_to_panel(rootfind_panel,&root_find_method,root_find_method,control_cb);
	glui->add_radiobutton_to_group(group3,"BezierClipping");
	glui->add_radiobutton_to_group(group3,"Secant method");

	GLUI_Panel *objectkind_panel = glui->add_panel("Object");
	GLUI_RadioGroup *group4 = glui->add_radiogroup_to_panel(objectkind_panel,&kindOfObject,kindOfObject,init);
	glui->add_radiobutton_to_group(group4,"None");
	glui->add_radiobutton_to_group(group4,"RING");
	glui->add_radiobutton_to_group(group4,"CYLINDER");
	glui->add_radiobutton_to_group(group4,"LIGHT");

	GLUI_Panel *build_panel = glui->add_panel("Build");
	GLUI_RadioGroup *group5 = glui->add_radiogroup_to_panel(build_panel,&isSAH,isSAH,control_cb);
	glui->add_radiobutton_to_group(group5,"spacial meadian");
	glui->add_radiobutton_to_group(group5,"surface area herustic");

	GLUI_Panel *bvh_panel = glui->add_panel("Kind of BVH");
	GLUI_RadioGroup *group6 = glui->add_radiogroup_to_panel(bvh_panel,&kindOfBVH,kindOfBVH,control_cb);
	glui->add_radiobutton_to_group(group6,"Standard");
	glui->add_radiobutton_to_group(group6,"Modified");
	glui->add_radiobutton_to_group(group6,"Fitted");

	GLUI_Panel *prc_panel   = glui->add_panel("Kind of Processor");
	GLUI_RadioGroup *group7 = glui->add_radiogroup_to_panel(prc_panel,&isGPU,isGPU,control_cb);
	glui->add_radiobutton_to_group(group7,"CPU");
	glui->add_radiobutton_to_group(group7,"GPU");

	glui->add_button("Load from file",0,init_spheres_from_file);
	glui->add_button("Exit",0,gluiCallback);
	glui->set_main_gfx_window(main_window);
	glutMainLoop();

	delDevice();

	return 0;
}
