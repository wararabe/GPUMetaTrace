#ifndef __SVECTOR3_CUH__
#define __SVECTOR3_CUH__

typedef float3 SVector3;

__device__ inline SVector3 setVector(float e0, float e1, float e2){
    SVector3 temp = {e0, e1, e2};
    return temp;
}

__device__ inline SVector3 setVector(float e0){
    SVector3 temp = {e0, e0, e0};
    return temp;
}

__device__ inline SVector3 operator-(const SVector3& in_vec){
    SVector3 temp = {-in_vec.x, -in_vec.y, -in_vec.z};
    return temp;
}

__device__ inline float squaredLength(const SVector3& in_vec){
    return in_vec.x*in_vec.x + in_vec.y*in_vec.y + in_vec.x*in_vec.y;
}

__device__ inline float length(const SVector3& in_vec){
    return sqrt(squaredLength(in_vec));
}

__device__ inline bool operator<(const SVector3& v1, const SVector3& v2){
    return (v1.x < v2.x) && (v1.y < v2.y) && (v1.z < v2.z);
}

__device__ inline bool operator<=(const SVector3& v1, const SVector3& v2){
    return (v1.x <= v2.x) && (v1.y <= v2.y) && (v1.z <= v2.z);
}

__device__ inline bool operator>(const SVector3& v1, const SVector3& v2){
    return operator<(v2, v1);
}

__device__ inline bool operator>=(const SVector3& v1, const SVector3& v2){
    return operator<=(v2, v1);
}

__device__ inline SVector3 operator+(const SVector3& v1, const SVector3& v2){
    SVector3 temp = {v1.x+v2.x, v1.y+v2.y, v1.z+v2.z};
    return temp;
}

__device__ inline SVector3 operator-(const SVector3& v1, const SVector3& v2){
    SVector3 temp = {v1.x-v2.x, v1.y-v2.y, v1.z-v2.z};
    return temp;
}

__device__ inline SVector3 operator*(const SVector3& v1, const SVector3& v2){
    SVector3 temp = {v1.x*v2.x, v1.y*v2.y, v1.z*v2.z};
    return temp;
}

__device__ inline SVector3 operator*(const SVector3& v1, float t){
    SVector3 temp = {v1.x*t, v1.y*t, v1.z*t};
    return temp;
}

__device__ inline SVector3 operator*(float t, const SVector3& v2){
    SVector3 temp = {v2.x*t, v2.y*t, v2.z*t};
    return temp;
}

__device__ inline SVector3 operator/(const SVector3& v1, const SVector3& v2){
    SVector3 temp = {v1.x/v2.x, v1.y/v2.y, v1.z/v2.z};
    return temp;
}

__device__ inline SVector3 operator/(const SVector3& v1, float t){
    float inv_t = 1.0f/t;
    SVector3 temp = {v1.x*inv_t, v1.y*inv_t, v1.z*inv_t};
    return temp;
}

__device__ inline SVector3& operator+=(SVector3& vec, const SVector3& a){
    vec.x += a.x;
    vec.y += a.y;
    vec.z += a.z;
    return vec;
}

__device__ inline SVector3& operator-=(SVector3& vec, const SVector3& a){
    vec.x -= a.x;
    vec.y -= a.y;
    vec.z -= a.z;
    return vec;
}

__device__ inline SVector3& operator*=(SVector3& vec, const SVector3& a){
    vec.x *= a.x;
    vec.y *= a.y;
    vec.z *= a.z;
    return vec;
}

__device__ inline SVector3& operator*=(SVector3& vec, float t){
    vec.x *= t;
    vec.y *= t;
    vec.z *= t;
    return vec;
}

__device__ inline SVector3& operator/=(SVector3& vec, const SVector3& a){
    vec.x /= a.x;
    vec.y /= a.y;
    vec.z /= a.z;
    return vec;
}

__device__ inline SVector3& operator/=(SVector3& vec, float t){
    float inv_t = 1.0f / t;
    vec.x *= inv_t;
    vec.y *= inv_t;
    vec.z *= inv_t;
    return vec;
}

__device__ inline SVector3 unitVector(const SVector3& v){
    float len = length(v);
    return v / len;
}

__device__ inline SVector3 cross(const SVector3& v1, const SVector3& v2){
    SVector3 temp = {v1.y*v2.z-v1.z*v2.y, v1.z*v2.x-v1.x*v2.z, v1.x*v2.y-v1.y*v2.x};
    return temp;
}

__device__ inline float dot(const SVector3& v1, const SVector3& v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ inline float tripleProduct(const SVector3& v1, const SVector3& v2, const SVector3& v3){
    return dot((cross(v1, v2)), v3);
}

__device__ inline SVector3 absVector(const SVector3& v) {
    SVector3 temp = {fabs(v.x), fabs(v.y), fabs(v.z)};
    return temp;
}

#endif