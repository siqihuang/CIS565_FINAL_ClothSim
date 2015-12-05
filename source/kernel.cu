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
static glm::vec3 *angular_momentum,*dev_angular_momentum;
static glm::mat3x3 *inertia,*dev_inertia;
static float *dev_dist;
static int height,width,dimension,constraintNum,primitiveNum,triangleNum;
static float mass;
static float restitution_coefficient,damping_coefficient;
static int torn,*dev_torn;
static int *torn_id,*dev_torn_id;

static glm::vec3 * dev_k1_x;
static glm::vec3 * dev_k1_v;
static glm::vec3 * dev_k2_x;
static glm::vec3 * dev_k2_v;
static glm::vec3 * dev_k3_x;
static glm::vec3 * dev_k3_v;
static glm::vec3 * dev_k4_x;
static glm::vec3 * dev_k4_v;

static glm::vec3 * dev_pos_temp1,*pos_temp1;
static glm::vec3 * dev_vel_temp1,*vel_temp1;
static glm::vec3 * dev_external_force;

static float* dev_vel_implicit;
static float* dev_b_implicit;
static float* dev_force_implicit;
static int* dev_coo_Rows;
static int* dev_csr_Rows;
static int* dev_Cols;
static float* dev_Val;
static int dev_nnz;

static int springConstraintNum;

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

// implicit method Ax=b compute the rhs
__global__ void compute_b(float mass, float dt, float* _dev_v, float* _dev_force, float* b,int N)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index<N){
		b[index] = mass*_dev_v[index] + dt*_dev_force[index];
	}
}

__global__ void convert_2_implicit_data(glm::vec3* data_in, float* data_out, int N)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index<N){
		data_out[3 * index] = data_in[index].x;
		data_out[3 * index + 1] = data_in[index].y;
		data_out[3 * index + 2] = data_in[index].z;
	}
}

__global__ void inv_convert_2_implicit_data(float* data_in, glm::vec3* data_out, int N)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index<N){
		data_out[index].x = data_in[3 * index];
		data_out[index].y = data_in[3 * index + 1];
		data_out[index].z = data_in[3 * index + 2];
	
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

__device__ bool SphereIntersectionTest(glm::vec3 &p, glm::vec3 &normal,float radius, glm::vec3 center,float &dist)
{
	float COLLISION_EPSILON=1e-1;
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
	float COLLISION_EPSILON=1e-1;
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

__device__ bool insideBoxOnGPU(glm::vec3 pos,kdtree *tree){
	if(pos.x<=tree->xMax&&pos.x>=tree->xMin&&pos.y<=tree->yMax&&pos.y>=tree->yMin&&
		pos.z<=tree->zMax&&pos.z>=tree->zMin){
		return true;
	}
	else return false;
}

__device__ void getNearbyTrianglesOnGPU(glm::vec3 pos,kdtree *tree, int *list){
	int count=0,num=0,n=0;
	kdtree *kd[1000];
	kd[count++]=tree;
	while(count<1000&&n!=count&&num<180){
		kdtree *current=kd[n];
		if(insideBoxOnGPU(pos,current)){
			if(current->lc==nullptr&&current->rc==nullptr) list[num++]=current->index;
			else{
				kd[count++]=current->lc;
				if(count>=1000) break;
				kd[count++]=current->rc;
			}
		}
		n++;
	}
}

__device__ glm::vec3 getNormalOnGPU(glm::vec3 *m_positions,glm::vec3 *m_normals,int *m_indices, unsigned short TriangleIndex){
	glm::vec3 n1,n2,n3,v1,v2,v3,n,crossN,v12,v13;
	unsigned int index1,index2,index3;
	index1=m_indices[3*TriangleIndex];
	index2=m_indices[3*TriangleIndex+1];
	index3=m_indices[3*TriangleIndex+2];
	v1=m_positions[index1];v2=m_positions[index2];v3=m_positions[index3];
	n1=m_normals[index1];n2=m_normals[index2];n3=m_normals[index3];
	
	v12=v1-v2;v13=v1-v3;
	v12=glm::normalize(v12);v13=glm::normalize(v13);
	crossN=glm::cross(v12,v13);
	crossN=glm::normalize(crossN);
	
	n=(n1+n2+n3);
	n=glm::normalize(n);

	if(glm::dot(n,crossN)<0) return -crossN;
	else return crossN;
}

__device__ float getDistanceOnGPU(glm::vec3 *m_positions,glm::vec3 *m_normals,int *m_indices,glm::vec3 p,unsigned short TriangleIndex){
	float dis,k,x;
	unsigned short index;
	index=m_indices[3*TriangleIndex];
	
	glm::vec3 normal=getNormalOnGPU(m_positions,m_normals,m_indices,TriangleIndex);
	
	glm::vec3 d=p-m_positions[index];
	x=-(normal.x*d.x+normal.y*d.y+normal.z*d.z);
	//k=normal.x*normal.x+normal.y*normal.y+normal.z*normal.z;
	//dis=x/k;
	return x;
}

__device__ bool ObjectIntersectionTest(glm::vec3 & p, glm::vec3 & normal, kdtree *tree,glm::vec3 center,float &dist,
									   glm::vec3 *obj_vertex,glm::vec3 *obj_normal,int *obj_indices)
{
    // TODO
	float minDis=1e7;
	float COLLISION_EPSILON=1e-3;
	bool inCollision=false;
	glm::vec3 pos=p;//-center;
	int list[180];
	for(int i=0;i<180;++i) list[i]=-1;
	getNearbyTrianglesOnGPU(pos,tree,list);
	pos-=center;

	for(int i=0;i<180;i++){
		if(list[i]==-1) break;
		float tmp=getDistanceOnGPU(obj_vertex,obj_normal,obj_indices,pos,list[i]);
		if(tmp>0&&tmp<minDis&&tmp<0.1){
			glm::vec3 n=getNormalOnGPU(obj_vertex,obj_normal,obj_indices,list[i]);
			normal=n;
			minDis=tmp;
			inCollision=true;
		}
	}
	dist=-minDis-COLLISION_EPSILON;
	return inCollision;
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
			else if(primitive[i].type==3){
				if(ObjectIntersectionTest(pos[index],n,primitive[i].tree,primitive[i].pos,d,primitive[i].objVertex,primitive[i].objNormal,primitive[i].objIndices)){
					if (d < dist[index]){
						dist[index] = d;
						nor[index]= n;
					}
				}
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

__global__ void PBDProjectKernel(GPUConstraint *constraint,glm::vec3 *p,int *torn,int *torn_id,int N,int ns){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		if(constraint[index].type==0&&constraint[index].active){//Attachment Constraint
			float k_prime=1.0-pow(1.0-constraint[index].stiffnessPBD,1.0/ns);
			glm::vec3 v=p[constraint[index].fix_index];
			glm::vec3 dp=constraint[index].fixedPoint-v;
			
			atomicAdd(&p[constraint[index].fix_index].x,k_prime*dp.x);
			atomicAdd(&p[constraint[index].fix_index].y,k_prime*dp.y);
			atomicAdd(&p[constraint[index].fix_index].z,k_prime*dp.z);
			//p[constraint[index].fix_index]+=k_prime*dp;
		}
		else if(constraint[index].type==1&&constraint[index].active){//Spring Constraint
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

			if(current_length>1.2*rest_length){ 
				torn[0]=1;
				constraint[index].active=false;
				torn_id[constraint[index].triangleId1]=1;
				torn_id[constraint[index].triangleId2]=1;
			}
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
		if (dev_constraint[index].type == 0&&dev_constraint[index].active) //attachment constraint 
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
	cudaMalloc(&dev_constraint,(constraintNum+100)*sizeof(GPUConstraint));//give 100 more space for additional attachment constraint
	cudaMalloc(&dev_primitive,primitiveNum*sizeof(GPUPrimitive));
	cudaMalloc(&dev_collisionNormal,dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_dist,dimension*sizeof(float));
	cudaMalloc(&dev_angular_momentum,dimension*sizeof(float));
	cudaMalloc(&dev_inertia,dimension*sizeof(glm::mat3x3));

	cudaMalloc(&dev_k1_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k1_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k2_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k2_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k3_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k3_v, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k4_x, dimension*sizeof(glm::vec3));
	cudaMalloc(&dev_k4_v, dimension*sizeof(glm::vec3));

	cudaMalloc(&dev_vel_implicit, 3 * dimension*sizeof(float));
	cudaMalloc(&dev_force_implicit, 3 * dimension*sizeof(float));
	cudaMalloc(&dev_b_implicit, 3 * dimension*sizeof(float));

	cudaMalloc(&dev_torn_id,triangleNum*sizeof(int));
	cudaMalloc(&dev_torn,sizeof(int));

	cudaMemcpy(dev_pos,pos,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vel,vel,dimension*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_constraint,constraint,constraintNum*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_primitive,primitive,primitiveNum*sizeof(GPUPrimitive),cudaMemcpyHostToDevice);
	cudaMemset(dev_force,0,dimension*sizeof(glm::vec3));
	cudaMemset(dev_external_force,0,dimension*sizeof(glm::vec3));
	cudaMemset(dev_torn_id,0,triangleNum*sizeof(int));
	cudaMemset(dev_torn,0,sizeof(int));
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
	cudaFree(dev_angular_momentum);
	cudaFree(dev_inertia);

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

	cudaFree(dev_torn_id);
	cudaFree(dev_torn);

	delete(force);
	delete(pos);
	delete(vel);
	delete(pos_temp1);
	delete(vel_temp1);
	delete(angular_momentum);
	delete(inertia);
	delete(torn_id);
}

void copyData(GPUConstraint *GConstraint,GPUPrimitive *GPrimitive,glm::vec3 *Gpos,glm::vec3 *Gvel,int Gheight,int Gwidth
			  ,int GconstraintNum,int GspringConstraintNum,int GprimitiveNum,int GtriangleNum,float Gmass,float Grestitution_coefficient,float Gdamping_coefficient){
	constraint=GConstraint;
	primitive=GPrimitive;
	height=Gheight;
	width=Gwidth;
	dimension=height*width;
	constraintNum=GconstraintNum;
	springConstraintNum=GspringConstraintNum;
	primitiveNum=GprimitiveNum;
	triangleNum=GtriangleNum;
	mass=Gmass;
	restitution_coefficient=Grestitution_coefficient;
	damping_coefficient=Gdamping_coefficient;
	force=new glm::vec3[height*width];
	pos=new glm::vec3[height*width];
	vel=new glm::vec3[height*width];
	pos_temp1=new glm::vec3[height*width];
	vel_temp1=new glm::vec3[height*width];
	angular_momentum=new glm::vec3[height*width];
	inertia=new glm::mat3x3[height*width];
	torn_id=new int[triangleNum];
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
	//cout<<primitive[1].tree->index<<endl;
	//if(primitive[1].tree==nullptr) cout<<"null"<<endl;
	collisionDetectionKernel<<<(dimension+255)/256,256>>>(dev_pos,dev_collisionNormal,dev_dist,dev_primitive,primitiveNum,dimension);
}

void resolveCollisionOnGPU(){
	collisionSolvingOnGPU<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,dev_dist,dev_collisionNormal,restitution_coefficient,dimension);
	cudaMemcpy(pos,dev_pos,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
}

__global__ void calculateAngularMomentum(glm::vec3 *pos,glm::vec3 *vel,glm::vec3 tmpPos,glm::vec3 *angular_momentum,
										 glm::mat3x3 *inertia,float mass,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		glm::vec3 r=pos[index]-tmpPos;
		angular_momentum[index]+=mass*glm::cross(r,vel[index]);
		glm::mat3x3 r_mat(1);
		r_mat[0][1]=r.z;
		r_mat[0][2]=-r.y;
		r_mat[1][0]=-r.z;
		r_mat[1][2]=r.x;
		r_mat[2][0]=r.y;
		r_mat[2][1]=-r.x;
		inertia[index]=r_mat*glm::transpose(r_mat)*mass;
	}
}

__global__ void calculateVelocityDamping(glm::vec3 *pos,glm::vec3 *vel,glm::vec3 tmpPos,glm::vec3 tmpVel,
										 glm::vec3 angular_momentum,glm::mat3x3 inertia,float damping_coefficient,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		glm::vec3 r=pos[index]-tmpPos;
		glm::vec3 angular_vel=glm::inverse(inertia)*angular_momentum;
		glm::vec3 delta_v=tmpVel+glm::cross(angular_vel,r)-vel[index];
		vel[index]+=damping_coefficient*delta_v;
	}
}

void dampVelocityOnGPU()
{
	if (std::abs(damping_coefficient) < 1e-15) return;

	cudaMemcpy(pos_temp1,dev_pos,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
	cudaMemcpy(vel_temp1,dev_vel,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
	thrust::inclusive_scan(pos_temp1,pos_temp1+dimension,pos_temp1);
	thrust::inclusive_scan(vel_temp1,vel_temp1+dimension,vel_temp1);
	
	calculateAngularMomentum<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,pos_temp1[dimension-1]/(1.0f*dimension),dev_angular_momentum,
		dev_inertia,mass,dimension);
	cudaMemcpy(angular_momentum,dev_angular_momentum,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
	cudaMemcpy(inertia,dev_inertia,dimension*sizeof(glm::mat3x3),cudaMemcpyDeviceToHost);
	thrust::inclusive_scan(angular_momentum,angular_momentum+dimension,angular_momentum);
	thrust::inclusive_scan(inertia,inertia+dimension,inertia);
	
	calculateVelocityDamping<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,pos_temp1[dimension-1]/(1.0f*dimension),vel_temp1[dimension-1]/(1.0f*dimension),
		angular_momentum[dimension-1],inertia[dimension-1],damping_coefficient,dimension);
		
	cudaMemcpy(pos,dev_pos,dimension*sizeof(glm::vec3),cudaMemcpyDeviceToHost);
}

void integratePBDOnGPU(int ns,float dt)
{
	for(int i=0;i<ns;++i){
		vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_vel,dev_external_force,dev_vel,dt*1.0/mass,dimension);
		vector_add_mulvector<<<(dimension+255)/256,256>>>(dev_pos,dev_vel,dev_pbd,dt,dimension);

		PBDProjectKernel<<<(constraintNum+255)/256,256>>>(dev_constraint,dev_pbd,dev_torn,dev_torn_id,constraintNum,ns);
	
		vector_minus_vector_mul<<<(dimension+255)/256,256>>>(dev_pbd,dev_pos,dev_vel,1.0/(dt),dimension);
		vector_copy_vector<<<(dimension+255)/256,256>>>(dev_pos,dev_pbd,dimension);
		detectCollisionOnGPU();
		resolveCollisionOnGPU();
	}
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

void integrateImplicitBW_GPU(float dt)
{
	cudaMemset(dev_force, 0, dimension*sizeof(glm::vec3));

	//compute force
	kern_compute_force << <(constraintNum + 255) / 256, 256 >> >(dev_force, dev_pos, dev_constraint, constraintNum);
	cudaDeviceSynchronize();
	//add external force 
	vector_add_vector << <(dimension + 255) / 256, 256 >> > (dev_force, dev_external_force, dev_force, dimension);
    
	//convert to implicit data
	convert_2_implicit_data << <(dimension + 255) / 256, 256 >> > (dev_vel,dev_vel_implicit,dimension);
	convert_2_implicit_data << <(dimension + 255) / 256, 256 >> > (dev_force, dev_force_implicit, dimension);

	compute_b << <(3 * dimension + 255) / 256, 256 >> > (mass, dt, dev_vel_implicit, dev_force_implicit, dev_b_implicit, 3 * dimension);

	// --- create library handles:
	cusolverSpHandle_t cusolver_handle;
	cusolverStatus_t cusolver_status;
	cusolver_status = cusolverSpCreate(&cusolver_handle);
	//std::cout << "status create cusolver handle: " << cusolver_status << std::endl;

	cusparseHandle_t cusparse_handle;
	cusparseStatus_t cusparse_status;
	cusparse_status = cusparseCreate(&cusparse_handle);
	//std::cout << "status create cusparse handle: " << cusparse_status << std::endl;

	
	
	cusparseMatDescr_t descrA;

	cusparse_status = cusparseCreateMatDescr(&descrA);
	//std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;

	//solving
	
	float tol = 1e-3;
	int reorder = 0;
	int singularity = 0;
	
	//std::cout << dev_nnz << std::endl;
	
	cusolver_status = cusolverSpScsrlsvchol(cusolver_handle, 3 * dimension, dev_nnz, descrA, dev_Val,
		dev_csr_Rows, dev_Cols, dev_b_implicit, tol, reorder, dev_vel_implicit,
		&singularity);

	cudaDeviceSynchronize();

	//std::cout << "singularity (should be -1): " << singularity << std::endl;

	//std::cout << "status cusolver solving (!): " << cusolver_status << std::endl;

	// relocated these 2 lines from above to solve (2):
	cusparse_status = cusparseDestroy(cusparse_handle);
	//std::cout << "status destroy cusparse handle: " << cusparse_status << std::endl;

	cusolver_status = cusolverSpDestroy(cusolver_handle);
	//std::cout << "status destroy cusolver handle: " << cusolver_status << std::endl;

	//convert the data back
	inv_convert_2_implicit_data << <(dimension + 255) / 256, 256 >> > (dev_vel_implicit, dev_vel, dimension);

	vector_add_mulvector << <(dimension + 255) / 256, 256 >> >(dev_pos, dev_vel, dev_pos, dt, dimension);

}

//====================	integration	====================

kdtree *initTree(kdtree *root){
	//postorder method to first get the left and right child on GPU Memory, then replace it with the memory on CPU, then copy the whole point to GPU
	if(root==nullptr) return nullptr;
	kdtree *dev_lc=initTree(root->lc);
	kdtree *dev_rc=initTree(root->rc);
	kdtree *tmp=new kdtree(root);
	tmp->lc=dev_lc;
	tmp->rc=dev_rc;
	kdtree *dev_root;
	cudaMalloc(&dev_root,sizeof(kdtree));
	cudaMemcpy(dev_root,tmp,sizeof(kdtree),cudaMemcpyHostToDevice);
	return dev_root;
}

void updateAttachmentConstraintOnGPU(GPUConstraint *Gconstraint,int n){
	cudaMemset(dev_constraint+springConstraintNum,0,100*sizeof(GPUConstraint));
	n=min(100,n);//no more than 100 Attachment Constraint
	cudaMemcpy(dev_constraint+springConstraintNum,Gconstraint,n*sizeof(GPUConstraint),cudaMemcpyHostToDevice);
	constraintNum=springConstraintNum+n;
}

void convertSystemMatrix(std::vector<int> &host_Rows, std::vector<int> &host_Cols, std::vector<float> &host_Val)
{
	//step1 convert to coo format

	int nnz = host_Val.size();
	dev_nnz = nnz;

	cudaMalloc((void**)&dev_Val, nnz*sizeof(float));
	cudaMalloc((void**)&dev_coo_Rows, nnz*sizeof(int));
	cudaMalloc((void**)&dev_csr_Rows, (dimension*3 + 1)*sizeof(int));
	cudaMalloc((void**)&dev_Cols, nnz*sizeof(int));




	cudaMemcpy(dev_Val, host_Val.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_coo_Rows, host_Rows.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Cols, host_Cols.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);


	//std::vector<float> hst_rows(nnz, 0);

	//cudaMemcpy(hst_rows.data(), dev_Val, nnz*sizeof(float), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < 10; i++)
	//{
	//	std::cout << hst_rows[i] << std::endl;
	//}

	cusparseHandle_t cusparse_handle;
	cusparseStatus_t cusparse_status;
	cusparse_status = cusparseCreate(&cusparse_handle);
	std::cout << "status create cusparse handle: " << cusparse_status << std::endl;


	cusparse_status = cusparseXcoo2csr(cusparse_handle, dev_coo_Rows, nnz, dimension*3,
		dev_csr_Rows, CUSPARSE_INDEX_BASE_ZERO);
	std::cout << "status cusparse coo2csr conversion: " << cusparse_status << std::endl;

	cudaDeviceSynchronize(); // matrix format conversion has to be finished!
	

	
	//check the matrix
	
	//cusparseMatDescr_t descrA;

	//cusparse_status = cusparseCreateMatDescr(&descrA);
	//std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;
	//
	//std::vector<float> A(dimension * 3 * dimension * 3, 0);
	//float *dA;
	//cudaMalloc((void**)&dA, A.size()*sizeof(float));

	//cusparseScsr2dense(cusparse_handle, dimension * 3, dimension * 3, descrA, dev_Val,
	//	dev_csr_Rows, dev_Cols, dA, dimension * 3);

	//cudaMemcpy(A.data(), dA, A.size()*sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "A: \n";
	//for (int i = 0; i < 10; ++i) {
	//	for (int j = 0; j < 10; ++j) {
	//		std::cout << A[i*dimension * 3 + j] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	//cudaFree(dA);


	cusparse_status = cusparseDestroy(cusparse_handle);
	std::cout << "status destroy cusparse handle: " << cusparse_status << std::endl;
}

glm::vec3 *getPos(){
	return pos;
}

glm::vec3 *getVel(){
	return vel;
}

bool isTorn(){
	cudaMemcpy(&torn,dev_torn,sizeof(int),cudaMemcpyDeviceToHost);
	if(torn==1){ 
		cudaMemset(dev_torn,0,sizeof(int));
		return true;
	}
	return false;
}

int *getTornId(){
	cudaMemcpy(torn_id,dev_torn_id,triangleNum*sizeof(int),cudaMemcpyDeviceToHost);
	return torn_id;
}

void resetTornFlag(){
	torn=0;
	cudaMemset(dev_torn,0,sizeof(int));
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