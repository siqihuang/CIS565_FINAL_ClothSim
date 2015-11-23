#include "cuda.h"
#include <iostream>
#include "kernel.h"

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

void initData(){
	cudaMalloc(&dev_pos,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_vel,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_force,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_pbd,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_constraint,constraintNum*sizeof(GPUConstraint));
	cudaMalloc(&dev_primitive,primitiveNum*sizeof(GPUPrimitive));
	cudaMalloc(&dev_collisionNormal,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_dist,dimension*sizeof(float));

	cudaMemcpy(dev_pos,pos,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vel,vel,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_constraint,constraint,constraintNum*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_primitive,primitive,primitiveNum*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	cudaMemset(dev_force,0,dimension*sizeof(glm::vec3));
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

/*
test cuda core function
*/
__global__ void test(int *a,int *b,int *c,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		c[index]=a[index]+b[index];
	}
}

__device__ bool StaticIntersectionTest(glm::vec3 &p, glm::vec3 &normal, glm::vec3 dim,glm::vec3 center,float &dist)
{
	float COLLISION_EPSILON=1e-1;
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

__global__ void collisionDetectionKernel(glm::vec3 *pos,glm::vec3 *nor,float *dist,GPUPrimitive *primitive,int primitiveNum,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		dist[index]=0;
		nor[index]=glm::vec3(0);
		for(int i=0;i<primitiveNum;++i){
			if(primitive[i].type==0){
				float d=0;
				glm::vec3 n(0,1,0);
				if(StaticIntersectionTest(pos[index],n,primitive[i].cSize,primitive[i].pos,d)){
					if (d < dist[index]){
						dist[index] = d;
						nor[index]= n;
					}
				}//if
			}//if
			else if(primitive[i].type==1){}
		}
	}
}

void detectCollisionOnGPU(){
	cudaMemset(dev_collisionNormal,0,dimension*sizeof(glm::vec3));//reset every time,may need stream compaction to improve
	cudaMemset(dev_dist,0,dimension*sizeof(float));
	collisionDetectionKernel<<<(dimension+255)/256,256>>>(dev_pos,dev_collisionNormal,dev_dist,dev_primitive,primitiveNum,dimension);
}

__global__ void collisionSolvingOnGPU(glm::vec3 *pos,glm::vec3 *vel,float *dist,glm::vec3 *nor,float restitution_coefficient,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N&&glm::length(nor[index])>0.1){
		pos[index]-=nor[index]*dist[index];
		float n=glm::dot(nor[index],vel[index]);
		vel[index]+=-(1+restitution_coefficient)*n*nor[index];
	}
}

void resolveCollisionOnGPU(){
	collisionSolvingOnGPU<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,dev_dist,dev_collisionNormal,restitution_coefficient,dimension);
	cudaMemcpy(pos,dev_pos,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
	//cudaMemcpy(vel,dev_vel,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
}

__global__ void addGravityOnGPU(glm::vec3 *force,float mass,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		force[index].y=-9.80f*mass;
		force[index].x=0;
		force[index].z=0;
	}
}

__global__ void vector_copy_vector(glm::vec3 *v1,glm::vec3 *v2,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v1[index]=v2[index];
	}
}

__global__ void vector_add_vector(glm::vec3 *v1,glm::vec3 *v2,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v1[index]+=v2[index];
	}
}

__global__ void vector_minus_vector_mul(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v1[index]=(v2[index]-v3[index])*mul;
	}
}

__global__ void vector_add_mulvector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		//v1[index]=glm::vec3(0);
		v1[index]=v2[index]+mul*v3[index];
	}
}

__global__ void vector_minus_vector(glm::vec3 *v1,glm::vec3 *v2,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		v1[index]-=v2[index];
	}
}

__global__ void vector_minus_mulvector(glm::vec3 *v1,glm::vec3 *v2,glm::vec3 *v3,float mul,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		if(glm::length(v3[index])<1e-3) v1[index]=v2[index];
		else v1[index]=v2[index]-mul*v3[index];
	}
}

void calculateExternalForceoOnGPU()
{
	addGravityOnGPU<<<(dimension+255)/256,256>>>(dev_force,mass,dimension);
}

__global__ void PBDProjectKernel(GPUConstraint *constraint,glm::vec3 *p,int N,int ns){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		if(constraint[index].type==0){//Attachment Constraint
		}
		else{//Spring Constraint
			float k_prime=1.0-pow(1.0-constraint[index].stiffnessPBD,1.0/ns);
			float rest_length=constraint[index].length;
			glm::vec3 v1=p[constraint[index].p1];
			glm::vec3 v2=p[constraint[index].p2];
			float current_length=glm::length(v1-v2);
			glm::vec3 current_direction=(v1-v2)/current_length;
			glm::vec3 dp=(current_length-rest_length)*current_direction;

			//atomicAdd(&p[constraint[index].p1],-dp*0.5f*k_prime);
			//atomicAdd(&p[constraint[index].p2],dp*0.5f*k_prime);
			p[constraint[index].p1]-=0.5f*k_prime*dp;
			p[constraint[index].p2]+=0.5f*k_prime*dp;
		}

	}
}

void integratePBDOnGPU(unsigned int ns,float dt)
{
	float t=ns*dt/mass;
	vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_vel,dev_vel,dev_force,t,dimension);
	vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_pbd,dev_pos,dev_vel,ns*dt,dimension);
	
	PBDProjectKernel<<<(constraintNum+255)/256,256>>>(dev_constraint,dev_pbd,constraintNum,ns);
	
	vector_minus_vector_mul<<<(dimension+255)/256,256>>>(dev_vel,dev_pbd,dev_pos,1.0/dt/ns,dimension);
	vector_copy_vector<<<(dimension+255)/256,256>>>(dev_pos,dev_pbd,dimension);
}

glm::vec3 *getPos(){
	return pos;
}

glm::vec3 *getVel(){
	return vel;
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