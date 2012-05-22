#ifndef __CAMERA_CUH__
#define __CAMERA_CUH__



typedef struct Camera{

	SVector3 m_EyePoint;

	float m_FocalLength;
	float m_FocusLength;
	float m_RoundAngle;
	float m_HorizontalAngle;
	float m_VerticalAngle;
	float m_IrisF;

	SVector3 p1;
	SVector3 p2;
	SVector3 p3;
	SVector3 p4;

	SVector3 camera_x;
	SVector3 camera_y;
	SVector3 camera_z;

	float m_T_Focus_p_i;

	float m_LensRadius;
	float m_T_i;
	float m_T_Lr_ips;
	float m_T_Lr_i;
	float m_Jitter;

	int m_Width;
	int m_Height;

	float m_T_InvWidth;
	float m_T_InvHeight;

}Camera;

__device__ Camera camera;

__device__ void cameraScreenView(const int x,const int y, SRay *ray){

	const SVector3 temp1 = camera.p1 * float(x) * camera.m_T_InvWidth + camera.p2 * (camera.m_Width - float(x)) * camera.m_T_InvWidth;
	const SVector3 temp2 = camera.p3 * float(x) * camera.m_T_InvWidth + camera.p4 * (camera.m_Width - float(x)) * camera.m_T_InvWidth;
	const SVector3 temp3 = temp1 * float(y) * camera.m_T_InvHeight + temp2 * (camera.m_Height - float(y)) * camera.m_T_InvHeight;

	const SVector3 dir = unitVector(camera.m_EyePoint - temp3);
	ray->direction.x = dir.x; ray->direction.y = dir.y; ray->direction.z = dir.z;
	ray->origin.x    = camera.m_EyePoint.x; ray->origin.y = camera.m_EyePoint.y; ray->origin.z = camera.m_EyePoint.z;

}

#endif