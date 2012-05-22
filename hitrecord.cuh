#ifndef __HITRECORD_CUH__
#define __HITRECORD_CUH__

typedef struct _SHitRecord{
	float ray_t;
	float min_t;
	float rD;
	int primitive_id;
}SHitRecord;



#endif