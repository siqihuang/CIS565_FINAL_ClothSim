// ---------------------------------------------------------------------------------//
// Copyright (c) 2013, Regents of the University of Pennsylvania                    //
// All rights reserved.                                                             //
//                                                                                  //
// Redistribution and use in source and binary forms, with or without               //
// modification, are permitted provided that the following conditions are met:      //
//     * Redistributions of source code must retain the above copyright             //
//       notice, this list of conditions and the following disclaimer.              //
//     * Redistributions in binary form must reproduce the above copyright          //
//       notice, this list of conditions and the following disclaimer in the        //
//       documentation and/or other materials provided with the distribution.       //
//     * Neither the name of the <organization> nor the                             //
//       names of its contributors may be used to endorse or promote products       //
//       derived from this software without specific prior written permission.      //
//                                                                                  //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND  //
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED    //
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE           //
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY               //
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       //
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     //
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND      //
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT       //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS    //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                     //
//                                                                                  //
// Contact Tiantian Liu (ltt1598@gmail.com) if you have any questions.              //
//----------------------------------------------------------------------------------//

#ifndef _SIMULATION_H_
#define _SIMULATION_H_

#include <vector>

#include "global_headers.h"
#include "anttweakbar_wrapper.h"
#include "mesh.h"
#include "constraint.h"
#include "scene.h"
#include "kernel.h"

//----------cuda solver test--------------//
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cassert>



class Mesh;
class AntTweakBarWrapper;

extern bool g_GPU_render;

typedef enum
{
    INTEGRATION_EXPLICIT_EULER,
	INTEGRATION_EXPLICIT_RK2,
	INTEGRATION_EXPLICIT_RK4,
    INTEGRATION_EXPLICIT_SYMPLECTIC,
    INTEGRATION_IMPLICIT_EULER_BARAFF_WITKIN,
	INTEGRATION_POSITION_BASED_DYNAMICS,
    INTEGRATION_TOTAL_NUM

} IntegrationMethod;

struct CollisionInstance
{
	unsigned int point_index; // the id of the colliding particle
	EigenVector3 normal; // normal direction of the collider
	ScalarType dist; // dist is the distance of a particle penetrating into a collider,  dist < 0 

	CollisionInstance(const unsigned int id, const EigenVector3& n, const ScalarType d) {point_index=id, normal=n, dist=d;}
};

class Simulation
{
    friend class AntTweakBarWrapper;

public:
    Simulation();
    virtual ~Simulation();

    void Reset();
    void Update();
	void GPUUpdate();
	void CPUUpdate();
    void DrawConstraints(const VBO& vbos);

    // select/unselect/move attachment constratins
    ScalarType TryToSelectAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir); // return ray_projection_plane_distance if hit; return -1 otherwise.
    bool TryToToggleAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir); // true if hit some vertex/constraint
    void SelectAtttachmentConstraint(AttachmentConstraint* ac);
    void UnselectAttachmentConstraint();
    void AddAttachmentConstraint(unsigned int vertex_index); // add one attachment constraint at vertex_index
    void MoveSelectedAttachmentConstraintTo(const EigenVector3& target); // move selected attachement constraint to target

    inline void SetMesh(Mesh* mesh) {m_mesh = mesh;}
    inline void SetScene(Scene* scene) {m_scene = scene;}
    
protected:

    // simulation constants
    ScalarType m_h; // time_step

    // simulation constants
    ScalarType m_gravity_constant;
    ScalarType m_stiffness_attachment;
    ScalarType m_stiffness_stretch;
    ScalarType m_stiffness_bending;
	ScalarType m_stiffness_attachment_pbd;
	ScalarType m_stiffness_stretch_pbd;
	ScalarType m_stiffness_bending_pbd;
    ScalarType m_damping_coefficient;
	ScalarType m_restitution_coefficient;

    // integration method
    IntegrationMethod m_integration_method;

    // key simulation components: mesh and scene
    Mesh *m_mesh;
    Scene *m_scene;
    // key simulation components: constraints
    std::vector<Constraint*> m_constraints;
    AttachmentConstraint* m_selected_attachment_constraint;

    // external force (gravity, wind, etc...)
    VectorX m_external_force;

    // number of iterations per frame
    unsigned int m_iterations_per_frame;


	//for implicit method
	SparseMatrix m_A;
	int m_nnz;

private:

    // main update sub-routines
    void clearConstraints(); // cleanup all constraints
    void setupConstraints(); // initialize constraints
    void dampVelocity(); // damp velocity at the end of each iteration.
    void calculateExternalForce();
    void detectCollision(const VectorX& x, std::vector<CollisionInstance>& collisions); // detect collision and return a vector of penetration
	void resolveCollision(VectorX&x, VectorX& v, std::vector<CollisionInstance>& collisions);

	void integrateExplicitEuler(VectorX& x, VectorX& v, ScalarType dt);
	void integrateExplicitRK2(VectorX& x, VectorX& v, ScalarType dt);
	void integrateExplicitRK4(VectorX& x, VectorX& v, ScalarType dt);
    void integrateExplicitSymplectic(VectorX& x, VectorX& v, ScalarType dt);
	void integrateImplicitBW(VectorX& x, VectorX& v, ScalarType dt);
	void integratePBD(VectorX& x, VectorX& v, unsigned int ns);

    // for explicit/implicit integration only
    void computeForces(const VectorX& x, VectorX& force);
    void computeStiffnessMatrix(const VectorX& x, SparseMatrix& stiffness_matrix);

    // utility functions
    void factorizeDirectSolverLLT(const SparseMatrix& A, Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>& lltSolver, char* warning_msg = ""); // factorize matrix A using LLT decomposition
    void factorizeDirectSolverLDLT(const SparseMatrix& A, Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>& ldltSolver, char* warning_msg = ""); // factorize matrix A using LDLT decomposition
    void generateRandomVector(const unsigned int size, VectorX& x); // generate random vector varing from [-1 1].
	
	//GPU version
	void copyDataToGPU();
	void computeSystemMatrix(); //A = M-dt*dt*K; K is stiffness matrix
};

#endif