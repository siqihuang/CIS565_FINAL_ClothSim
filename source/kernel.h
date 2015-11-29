#pragma once
#ifndef _KERNEL_
#define _KERNEL_
#include <cuda.h>
#include<thrust/scan.h>
#include <vector>
#include <string>
#include "glm.hpp"
#include "kdtree.h"

//----------cuda solver test--------------//
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cassert>

struct GPUConstraint{
	int type;//0 for Attachment constraint, 1 for Spring constraint
	float stiffness;
	float stiffnessPBD;
	int p1;
	int p2;
	int fix_index;
	float rest_length;
	glm::vec3 fixedPoint;
	bool active;
};

struct GPUPrimitive{
	int type;//0 for cube, 1 for sphere, 2 for plane, 3 for obj
	glm::vec3 pos;
	glm::vec3 cSize;//cube size
	float radius;
	glm::vec3 pNormal;//plane normal
	kdtree *tree;
	glm::vec3 *objVertex;
	glm::vec3 *objNormal;
	int *objIndices;
};

/*variables
m_mass_matrix
m_current_position
m_current_velocity
m_vertices_num
m_system_dimension
m_iterations_per_frame
m_h
*/

void testCuda();
void collisionResolving();
void copyData(GPUConstraint *GConstraint,GPUPrimitive *Gprimitive,glm::vec3 *pos,glm::vec3 *vel,int height,int width,
			  int constraintNum,int springConstraintNum,int primitiveNum,float mass,float restitution_coefficient,float damping_coefficient);
void calculateExternalForceoOnGPU();
void calculateExternalForceoOnGPU1();
void integratePBDOnGPU(int ns,float dt);
void initData();
void deleteData();
void detectCollisionOnGPU();
void resolveCollisionOnGPU();
void dampVelocityOnGPU();
void updateAttachmentConstraintOnGPU(GPUConstraint *Gconstraint,int n);

void integrateExplicitEuler_GPU(float dt);
void integrateExplicitRK2_GPU(float dt);
void integrateExplicitRK4_GPU(float dt);

<<<<<<< HEAD
void integrateImplicitBW_GPU(float dt);
void convertSystemMatrix(std::vector<int> & host_Rows, std::vector<int> & host_Cols, std::vector<float> & host_Val);
=======
kdtree *initTree(kdtree *root);
>>>>>>> origin/master

glm::vec3 *getPos();
glm::vec3 *getVel();







#endif