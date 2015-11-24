#include "cuda.h"
#include <iostream>
#include "kernel.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
			system("pause");\
			exit(1); \
		        } \
	    } while (0)


static GPUConstraint *constraint,*dev_constraint;
static GPUPrimitive *primitive,*dev_primitive;
static glm::vec3 *pos,*dev_pos;
static glm::vec3 *vel,*dev_vel;
static glm::vec3 *force,*dev_force;
static glm::vec3 *dev_pbd;
static glm::vec3 *dev_collisionNormal;
static float *dev_dist;
static int height,width,dimension,constraintNum,primitiveNum;
static float mass;
static float restitution_coefficient;

static glm::vec3 * dev_k1_x;
static glm::vec3 * dev_k1_v;
static glm::vec3 * dev_k2_x;
static glm::vec3 * dev_k2_v;
static glm::vec3 * dev_k3_x;
static glm::vec3 * dev_k3_v;
static glm::vec3 * dev_k4_x;
static glm::vec3 * dev_k4_v;

static glm::vec3 * dev_pos_temp1;
static glm::vec3 * dev_vel_temp1;
static glm::vec3 * dev_external_force;

/*
helper function for matrix operation
*/

//matrix copy matrix v1=v2
__global__ void vector_copy_vector(glm::vec3 *v1,glm::vec3 *v2,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v1[index]=v2[index];
	}
}

//matrix add matrix v3=v1+v2
__global__ void vector_add_vector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3, int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=v1[index]+v2[index];
	}
}

//matrix add matrix times a factor v3=v1+v2*mul
__global__ void vector_add_mulvector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=v1[index]+mul*v2[index];
	}
}

//matrix minus matrix v3=v1-v2
__global__ void vector_minus_vector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=v1[index]-v2[index];
	}
}

//matrix minus matrix times a factor v3=v1-v2*mul
__global__ void vector_minus_mulvector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=v1[index]-mul*v2[index];
	}
}

//matrix times a factor v_out=v_in*mul
__global__ void vector_mul_scalar(glm::vec3 * v_out, glm::vec3 * v_in, float mul ,int N)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index<N){
		v_out[index] = mul * v_in[index];
	}
}

//matrix minus a matrix and times a factor v3=(v1-v2)*mul
__global__ void vector_add_vector_mul(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=(v1[index]+v2[index])*mul;
	}
}

//matrix add a matrix and times a factor v3=(v1+v2)*mul
__global__ void vector_minus_vector_mul(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v3[index]=(v1[index]-v2[index])*mul;
	}
}

/*
helper function for matrix operation
*/

/*
Intersection Test
*/
__device__ bool CubeIntersectionTest(glm::vec3 &p, glm::vec3 &normal, glm::vec3 dim,glm::vec3 center,float &dist)
{
	float COLLISION_EPSILON=2e-2;
    glm::vec3 diff = p - center;
	float xcollide,ycollide,zcollide;
	xcollide=(fabs(diff.x)-dim.x-COLLISION_EPSILON);
	ycollide=(fabs(diff.y)-dim.y-COLLISION_EPSILON);
	zcollide=(fabs(diff.z)-dim.z-COLLISION_EPSILON);
	if(xcollide<0&&ycollide<0&&zcollide<0){
		if(xcollide>=ycollide&&xcollide>=zcollide){
			if(diff.x>0){
				normal=glm::vec3(1,0,0);
			}
			else{
				normal=glm::vec3(-1,0,0);
			}
			dist=xcollide;
			return true;
		}
		else if(ycollide>=xcollide&&ycollide>=zcollide){
			if(diff.y>0){
				normal=glm::vec3(0,1,0);
			}
			else{
				normal=glm::vec3(0,-1,0);
			}
			dist=ycollide;
			return true;
		}
		else if(zcollide>=ycollide&&zcollide>=xcollide){
			if(diff.z>0){
				normal=glm::vec3(0,0,1);
			}
			else{
				normal=glm::vec3(0,0,-1);
			}
			dist=zcollide;
			return true;
		}
	}
	else
	{
		dist=1;
		return false;
	}
}

__device__ bool SphereIntersectionTest(glm::vec3 &p, glm::vec3 &normal,float radius, glm::vec3 center,float &dist)
{
	float COLLISION_EPSILON=2e-2;
    glm::vec3 diff = p - center;
	dist = glm::length(diff) - radius - COLLISION_EPSILON;
    if (dist < 0)
    {
		normal = glm::normalize(diff);
        return true;
    }
    else
    {
        return false;
    }
}

__device__ bool PlaneIntersectionTest(glm::vec3 &p, glm::vec3 &normal, glm::vec3 pNormal, glm::vec3 center,float &dist)
{
	float COLLISION_EPSILON=2e-2;
    float height = center[1];
    dist = p.y - height - COLLISION_EPSILON;
    normal=pNormal;

    if (dist < 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}
/*
Intetsection Test
*/

__global__ void collisionDetectionKernel(glm::vec3 *pos,glm::vec3 *nor,float *dist,GPUPrimitive *primitive,int primitiveNum,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		dist[index]=0;
		nor[index]=glm::vec3(0);
		for(int i=0;i<primitiveNum;++i){
			float d=0;
			glm::vec3 n(0);
			if(primitive[i].type==0){
				if(CubeIntersectionTest(pos[index],n,primitive[i].cSize,primitive[i].pos,d)){
					if (d < dist[index]){
						dist[index] = d;
						nor[index]= n;
					}
				}//if
			}//if
			else if(primitive[i].type==1){
				if(SphereIntersectionTest(pos[index],n,primitive[i].radius,primitive[i].pos,d)){
					if (d < dist[index]){
						dist[index] = d;
						nor[index]= n;
					}
				}//if
			}
			else if(primitive[i].type==2){
				if(PlaneIntersectionTest(pos[index],n,primitive[i].pNormal,primitive[i].pos,d)){
					if (d < dist[index]){
						dist[index] = d;
						nor[index]= n;
					}
				}//if
			}
		}
	}
}

__global__ void collisionSolvingOnGPU(glm::vec3 *pos,glm::vec3 *vel,float *dist,glm::vec3 *nor,float restitution_coefficient,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N&&glm::length(nor[index])>0.1){
		pos[index]-=nor[index]*dist[index];
		float n=glm::dot(nor[index],vel[index]);
		vel[index]+=-(1+restitution_coefficient)*n*nor[index];
	}
}

__global__ void addGravityOnGPU(glm::vec3 *force,float mass,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		force[index].y=-9.80f*mass;
		force[index].x=0;
		force[index].z=0;
	}
}

__global__ void PBDProjectKernel(GPUConstraint *constraint,glm::vec3 *p,int N,int ns){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		if(constraint[index].type==0){//Attachment Constraint
			float k_prime=1.0-pow(1.0-constraint[index].stiffnessPBD,1.0/ns);
			glm::vec3 v=p[constraint[index].fix_index];
			glm::vec3 dp=constraint[index].fixedPoint-v;
			
			atomicAdd(&p[constraint[index].fix_index].x,k_prime*dp.x);
			atomicAdd(&p[constraint[index].fix_index].y,k_prime*dp.y);
			atomicAdd(&p[constraint[index].fix_index].z,k_prime*dp.z);
			//p[constraint[index].fix_index]+=k_prime*dp;
		}
		else{//Spring Constraint
			float k_prime=1.0-pow(1.0-constraint[index].stiffnessPBD,1.0/ns);
			float rest_length=constraint[index].rest_length;
			glm::vec3 v1=p[constraint[index].p1];
			glm::vec3 v2=p[constraint[index].p2];
			float current_length=glm::length(v1-v2);
			glm::vec3 current_direction=(v1-v2)/current_length;
			glm::vec3 dp=(current_length-rest_length)*current_direction;

			atomicAdd(&p[constraint[index].p1].x,-0.5f*k_prime*dp.x);
			atomicAdd(&p[constraint[index].p1].y,-0.5f*k_prime*dp.y);
			atomicAdd(&p[constraint[index].p1].z,-0.5f*k_prime*dp.z);
			atomicAdd(&p[constraint[index].p2].x,0.5f*k_prime*dp.x);
			atomicAdd(&p[constraint[index].p2].y,0.5f*k_prime*dp.y);
			atomicAdd(&p[constraint[index].p2].z,0.5f*k_prime*dp.z);
			//p[constraint[index].p1]-=0.5f*k_prime*dp;
			//p[constraint[index].p2]+=0.5f*k_prime*dp;
		}

	}
}

__global__ void kern_compute_force(glm::vec3* dev_force, glm::vec3* dev_pos, GPUConstraint *dev_constraint, int Num_Constraint)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < Num_Constraint)
	{
		if (dev_constraint[index].type == 0) //attachment constraint 
		{
			glm::vec3 p0 = dev_constraint[index].fixedPoint;
			glm::vec3 p1 = dev_pos[dev_constraint[index].fix_index];

			float cur_len = glm::length(p1 - p0);
			float stiffness = dev_constraint[index].stiffness;
			glm::vec3 cur_force = stiffness*(p0 - p1);
			
			//// atomic add
			atomicAdd(&(dev_force[dev_constraint[index].fix_index].x), cur_force.x);
			atomicAdd(&(dev_force[dev_constraint[index].fix_index].y), cur_force.y);
			atomicAdd(&(dev_force[dev_constraint[index].fix_index].z), cur_force.z);
			
			//dev_force[dev_constraint[index].fix_index] += cur_force;
		}
		else if (dev_constraint[index].type == 1) //spring constraint
		{
			glm::vec3 p1 = dev_pos[dev_constraint[index].p1];
			glm::vec3 p2 = dev_pos[dev_constraint[index].p2];

			float cur_len = glm::length(p1 - p2);
			float stiffness = dev_constraint[index].stiffness;
			glm::vec3 cur_force = stiffness*(cur_len - dev_constraint[index].rest_length) / cur_len*(p2 - p1);

			//// atomic add
			atomicAdd(&(dev_force[dev_constraint[index].p1].x), cur_force.x);
			atomicAdd(&(dev_force[dev_constraint[index].p1].y), cur_force.y);
			atomicAdd(&(dev_force[dev_constraint[index].p1].z), cur_force.z);

			atomicAdd(&(dev_force[dev_constraint[index].p2].x), -cur_force.x);
			atomicAdd(&(dev_force[dev_constraint[index].p2].y), -cur_force.y);
			atomicAdd(&(dev_force[dev_constraint[index].p2].z), -cur_force.z);

			//dev_force[dev_constraint[index].p1] += cur_force;
			//dev_force[dev_constraint[index].p2] -= cur_force;
		}
	}
}


__global__	void kern_RK4_computation(glm::vec3 *dev_out, glm::vec3 *dev_k1, glm::vec3 *dev_k2, glm::vec3 *dev_k3, glm::vec3 *dev_k4, float a, int N)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (index < N)
	{
		dev_out[index] = dev_out[index] + a * (dev_k1[index] + 2.f* dev_k2[index] + 2.f * dev_k3[index] + dev_k4[index]);
	}
}

void initData(){
	cudaMalloc(&dev_pos,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_vel,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_pos_temp1, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_vel_temp1, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_force,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_external_force, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_pbd,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_constraint,constraintNum*sizeof(GPUConstraint));
	cudaMalloc(&dev_primitive,primitiveNum*sizeof(GPUPrimitive));
	cudaMalloc(&dev_collisionNormal,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_dist,dimension*sizeof(float));

	cudaMalloc(&dev_k1_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k1_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k2_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k2_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k3_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k3_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k4_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k4_v, dimension*sizeof(glm::vec3));

	cudaMemcpy(dev_pos,pos,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vel,vel,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_constraint,constraint,constraintNum*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_primitive,primitive,primitiveNum*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	cudaMemset(dev_force,0,dimension*sizeof(glm::vec3));
	cudaMemset(dev_external_force,0,dimension*sizeof(glm::vec3));
}

void deleteData(){
	cudaFree(dev_constraint);
	cudaFree(dev_primitive);
	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_force);
	cudaFree(dev_external_force);
	cudaFree(dev_pbd);
	cudaFree(dev_collisionNormal);
	cudaFree(dev_dist);

	cudaFree(dev_k1_x);
	cudaFree(dev_k1_v);
	cudaFree(dev_k2_x);
	cudaFree(dev_k2_v);
	cudaFree(dev_k3_x);
	cudaFree(dev_k3_v);
	cudaFree(dev_k4_x);
	cudaFree(dev_k4_v);

	cudaFree(dev_pos_temp1);
	cudaFree(dev_vel_temp1);
}

void copyData(GPUConstraint *GConstraint,GPUPrimitive *GPrimitive,glm::vec3 *Gpos,glm::vec3 *Gvel,int Gheight,int Gwidth
			  ,int GconstraintNum,int GprimitiveNum,float Gmass,float Grestitution_coefficient){
	constraint=GConstraint;
	primitive=GPrimitive;
	height=Gheight;
	width=Gwidth;
	dimension=height*width;
	constraintNum=GconstraintNum;
	primitiveNum=GprimitiveNum;
	mass=Gmass;
	restitution_coefficient=Grestitution_coefficient;
	force=new glm::vec3[height*width];
	pos=new glm::vec3[height*width];
	vel=new glm::vec3[height*width];
	for(int i=0;i<height*width;++i){
		force[i]=glm::vec3(0);
		pos[i]=Gpos[i];
		vel[i]=Gvel[i];
	}
	
	initData();
}

void calculateExternalForceoOnGPU()
{
	cudaMemset(dev_external_force,0,dimension*sizeof(glm::vec3));
	addGravityOnGPU << <(dimension + 255) / 256, 256 >> >(dev_external_force, mass, dimension);
}

void detectCollisionOnGPU(){
	cudaMemset(dev_collisionNormal,0,dimension*sizeof(glm::vec3));//reset every time,may need stream compaction to improve
	cudaMemset(dev_dist,0,dimension*sizeof(float));
	collisionDetectionKernel<<<(dimension+255)/256,256>>>(dev_pos,dev_collisionNormal,dev_dist,dev_primitive,primitiveNum,dimension);
}

void resolveCollisionOnGPU(){
	collisionSolvingOnGPU<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,dev_dist,dev_collisionNormal,restitution_coefficient,dimension);
	cudaMemcpy(pos,dev_pos,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
}

void integratePBDOnGPU(int ns,float dt)
{
	vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_vel,dev_external_force,dev_vel,dt*ns*1.0/mass,dimension);
	vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,dev_pbd,dt*ns,dimension);
	//for(int i=0;i<ns;++i){
		PBDProjectKernel<<<(constraintNum+255)/256,256>>>(dev_constraint,dev_pbd,constraintNum,ns);
	//}
	vector_minus_vector_mul<<<(dimension+255)/256,256>>>(dev_pbd,dev_pos,dev_vel,1.0/(ns*dt),dimension);
	vector_copy_vector<<<(dimension+255)/256,256>>>(dev_pos,dev_pbd,dimension);
}

//====================	integration	====================

void integrateExplicitEuler_GPU(float dt)
{
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	cudaCheckErrors("kernel fail");
	//add external force
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force,dev_external_force,dev_force,dimension);
	
	//pos
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_pos, dev_vel, dev_pos,dt,dimension );
	//vel
	
	float dt_inv_mass = dt / mass;
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_vel, dev_force, dev_vel, dt_inv_mass, dimension);
	//clear the force mem
	//cudaMemset(dev_force, 0, dimension);
	cudaMemcpy(pos, dev_pos, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void integrateExplicitRK2_GPU(float dt)
{
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	cudaCheckErrors("kernel fail");

	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);

	//pos
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_pos, dev_vel, dev_pos_temp1, dt, dimension);
	//vel
	float dt_inv_mass = dt / mass;
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_vel, dev_force, dev_vel_temp1, dt_inv_mass, dimension);

	cudaMemset(dev_force,0,dimension*sizeof(glm::vec3));
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos_temp1, dev_constraint, constraintNum);


	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);
	//pos
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_pos, dev_vel_temp1, dev_pos, dt, dimension);
	//vel
	vector_add_mulvector << <(dimension + 255) / 256, 256 >> > (dev_vel, dev_force, dev_vel, dt_inv_mass, dimension);


	//clear the force mem
	//cudaMemset(dev_force, 0, dimension);
	cudaMemcpy(pos, dev_pos, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void integrateExplicitRK4_GPU(float dt)
{
	float half_dt = dt / 2;
	
	float inv_mass = 1.f / mass;
	//step1
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);

	cudaMemcpy(dev_k1_x, dev_vel, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	vector_mul_scalar << <(constraintNum + 255) / 256, 256 >> >(dev_k1_v, dev_force, inv_mass, dimension);

	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_pos, dev_k1_x, dev_pos_temp1, half_dt, dimension);
	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_vel, dev_k1_v, dev_vel_temp1, half_dt, dimension);

	//step 2

	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos_temp1, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	
	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);

	cudaMemcpy(dev_k2_x, dev_vel_temp1, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	vector_mul_scalar << <(constraintNum + 255) / 256, 256 >> >(dev_k2_v, dev_force, inv_mass, dimension);

	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_pos, dev_k2_x, dev_pos_temp1, half_dt, dimension);
	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_vel, dev_k2_v, dev_vel_temp1, half_dt, dimension);


	//step3
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos_temp1, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);
	
	cudaMemcpy(dev_k3_x, dev_vel_temp1, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	vector_mul_scalar << <(constraintNum + 255) / 256, 256 >> >(dev_k3_v, dev_force, inv_mass, dimension);

	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_pos, dev_k2_x, dev_pos_temp1, dt, dimension);
	vector_add_mulvector << <(constraintNum + 255) / 256, 256 >> >(dev_vel, dev_k2_v, dev_vel_temp1, dt, dimension);

	//step4
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos_temp1, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force,dimension);

	cudaMemcpy(dev_k4_x, dev_vel_temp1, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	vector_mul_scalar << <(constraintNum + 255) / 256, 256 >> >(dev_k4_v, dev_force, inv_mass, dimension);

	
	//all together
	float a = dt / 6.f;
	kern_RK4_computation << <(constraintNum + 255) / 256, 256 >> >(dev_pos, dev_k1_x, dev_k2_x, dev_k3_x, dev_k4_x, a, dimension);
	kern_RK4_computation << <(constraintNum + 255) / 256, 256 >> >(dev_vel, dev_k1_v, dev_k2_v, dev_k3_v, dev_k4_v, a, dimension);

	cudaMemcpy(pos, dev_pos, dimension*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

//====================	integration	====================

glm::vec3 *getPos(){
	return pos;
}

glm::vec3 *getVel(){
	return vel;
}

/*
test cuda core function
*/
__global__ void test(int *a,int *b,int *c,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		c[index]=a[index]+b[index];
	}
}

/*
test function for cuda setup
*/
void testCuda(){
	int *a,*b,*c;
	int *dev_a,*dev_b,*dev_c;
	a=new int[10];
	b=new int[10];
	c=new int[10];
	for(int i=0;i<10;++i){
		a[i]=i;
		b[i]=10-i;
	}
	cudaMalloc(&dev_a,10*sizeof(int));
	cudaMalloc(&dev_b,10*sizeof(int));
	cudaMalloc(&dev_c,10*sizeof(int));
	cudaMemcpy(dev_a,a,10*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,10*sizeof(int),cudaMemcpyHostToDevice);
	test<<<1,256>>>(dev_a,dev_b,dev_c,10);
	cudaMemcpy(c,dev_c,10*sizeof(int),cudaMemcpyDeviceToHost);
	for(int i=0;i<10;++i){
		std::cout<<a[i]<<","<<b[i]<<","<<c[i]<<std::endl;
	}
}