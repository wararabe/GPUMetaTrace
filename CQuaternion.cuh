#ifndef __CQUATERNION_CUH__
#define __CQUATERNION_CUH__

#include"SVector3.cuh"

class CQuaternion{
public:
	SVector3 v;
	float t;

	__device__ inline CQuaternion::CQuaternion(): t(0){}
	__device__ inline CQuaternion::CQuaternion(float in_t, const SVector3& in_v): t(in_t), v(in_v){}
	__device__ inline CQuaternion::CQuaternion(const SVector3& in_v): t(0), v(in_v)	{}
	__device__ inline CQuaternion CQuaternion::operator*(const CQuaternion& in_q){
		CQuaternion _res;
		_res.t = t * in_q.t - dot(v, in_q.v);
		_res.v = in_q.v * t + v * in_q.t + cross(in_q.v, v);
		return _res;

	}

	__device__ inline int __M(int l, int m){
		return l*3+m;
	}

	__device__ inline void CQuaternion::toMatrix(float* out_Mat)
	{
		out_Mat[__M(0, 0)] = 1.0f - 2.0f * (v.y*v.y + v.z*v.z);
		out_Mat[__M(0, 1)] = 2.0f * (v.x*v.y - v.z*t);
		out_Mat[__M(0, 2)] = 2.0f * (v.x*v.z + v.y*t);

		out_Mat[__M(1, 0)] = 2.0f * (v.x*v.y + v.z*t);
		out_Mat[__M(1, 1)] = 1.0f - 2.0f * (v.x*v.x + v.z*v.z);
		out_Mat[__M(1, 2)] = 2.0f * (v.y*v.z - v.x*t);

		out_Mat[__M(2, 0)] = 2.0f * (v.x*v.z - v.y*t);
		out_Mat[__M(2, 1)] = 2.0f * (v.y*v.z + v.x*t);
		out_Mat[__M(2, 2)] = 1.0f - 2.0f * (v.x*v.x + v.y*v.y);
	}

	__device__ inline CQuaternion rotationQuaternion(const SVector3& in_axis, float angle)
	{
		float cos_a = cos(-angle*float(0.5));
		float sin_a = sin(-angle*float(0.5));
		SVector3 axis = unitVector(in_axis);

		return CQuaternion(cos_a, axis * sin_a);
	}

};

__device__ SVector3 rotateVector(const SVector3& in_v, const SVector3& in_axis, float angle)
{
	CQuaternion p = CQuaternion(in_v);
	SVector3 axis = unitVector(in_axis);
	float cos_a = cos(angle*float(0.5));
	float sin_a = sin(angle*float(0.5));
	float m_sin_a = -sin_a;
	CQuaternion q = CQuaternion(cos_a, axis * sin_a);
	CQuaternion r = CQuaternion(cos_a, axis * m_sin_a);
	return (r * p * q).v;
}

#endif