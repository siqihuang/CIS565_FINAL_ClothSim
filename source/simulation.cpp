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

#pragma warning( disable : 4996)

#include "simulation.h"
#include "timer_wrapper.h"

Simulation::Simulation()

{
}

Simulation::~Simulation()
{
    clearConstraints();
}

void Simulation::Reset()
{    
    m_external_force.resize(m_mesh->m_system_dimension);
	deleteData();

    setupConstraints();
	computeSystemMatrix();
	EigenVector3 v;

    m_selected_attachment_constraint = NULL;
}

void Simulation::Update(){
	if(g_GPU_render){ 
		GPUUpdate();
	}
	else{
		CPUUpdate();
	}
}

void Simulation::GPUUpdate(){
	calculateExternalForceoOnGPU();

	float dt = m_h / m_iterations_per_frame;
	
	
	
	for (unsigned int it = 0; it != m_iterations_per_frame; ++it)
	{
		switch (m_integration_method)
		{
		case INTEGRATION_EXPLICIT_EULER:
			integrateExplicitEuler_GPU(dt);
			break;
		case INTEGRATION_EXPLICIT_RK2:
			integrateExplicitRK2_GPU(dt);
			break;
		case INTEGRATION_EXPLICIT_RK4:
			integrateExplicitRK4_GPU(dt);
			break;
		case INTEGRATION_IMPLICIT_EULER_BARAFF_WITKIN:
			integrateImplicitBW_GPU(dt);
			break;
		}
	}


	// pbd integration method
	if (m_integration_method == INTEGRATION_POSITION_BASED_DYNAMICS)
	{
		
		integratePBDOnGPU(m_iterations_per_frame, dt);
	
	}

	detectCollisionOnGPU();
	resolveCollisionOnGPU();

	glm::vec3 *pos;
	pos=getPos();
	int size=m_mesh->m_dim[0]*m_mesh->m_dim[1];
	for(int i=0;i<size;++i){
		m_mesh->m_current_positions.block_vector(i)=GLM2Eigen(pos[i]);
	}
}

void Simulation::CPUUpdate()
{
	// LOOK: main simulation loop

    // update external force
    calculateExternalForce();

	// get a reference of current position and velocity
	VectorX& x = m_mesh->m_current_positions;
	VectorX& v = m_mesh->m_current_velocities;
	// substepping timestep for explicit/implicit euler
	ScalarType dt = m_h / m_iterations_per_frame;
	for (unsigned int it = 0; it != m_iterations_per_frame; ++it)
	{
		// update cloth
		switch (m_integration_method)
		{
		case INTEGRATION_EXPLICIT_EULER:
			integrateExplicitEuler(x, v, dt);
			break;
		case INTEGRATION_EXPLICIT_RK2:
			integrateExplicitRK2(x, v, dt);
			break;
		case INTEGRATION_EXPLICIT_RK4:
			integrateExplicitRK4(x, v, dt);
			break;
		case INTEGRATION_EXPLICIT_SYMPLECTIC:
			integrateExplicitSymplectic(x, v, dt);
			break;
		case INTEGRATION_IMPLICIT_EULER_BARAFF_WITKIN:
			integrateImplicitBW(x, v, dt);
			break;
		}
	}
	// pbd integration method
	if (m_integration_method == INTEGRATION_POSITION_BASED_DYNAMICS)
	{
		integratePBD(x, v, m_iterations_per_frame);
	}

	// collision detection and resolution
	std::vector<CollisionInstance> collisions;
	detectCollision(x, collisions);
	resolveCollision(x, v, collisions);

    // damping
    //dampVelocity();
}

void Simulation::DrawConstraints(const VBO& vbos)
{
    for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
    {
        (*it)->Draw(vbos);
    }
}

ScalarType Simulation::TryToSelectAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir)
{
    ScalarType ray_point_dist;
    ScalarType min_dist = 100.0;
    AttachmentConstraint* best_candidate = NULL;

    bool current_state_on = false;
    for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
    {
        AttachmentConstraint* ac;
        if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
        {
            ray_point_dist = ((ac->GetFixedPoint()-p0).cross(dir)).norm();
            if (ray_point_dist < min_dist)
            {
                min_dist = ray_point_dist;
                best_candidate = ac;
            }
        }
    }
    // exit if no one fits
    if (min_dist > DEFAULT_SELECTION_RADIUS)
    {
        UnselectAttachmentConstraint();

        return -1;
    }
    else
    {
        SelectAtttachmentConstraint(best_candidate);
        EigenVector3 fixed_point_temp = m_mesh->m_current_positions.block_vector(m_selected_attachment_constraint->GetConstrainedVertexIndex());

        return (fixed_point_temp-p0).dot(dir); // this is m_cached_projection_plane_distance
    }
}

bool Simulation::TryToToggleAttachmentConstraint(const EigenVector3& p0, const EigenVector3& dir)
{
    EigenVector3 p1;

    ScalarType ray_point_dist;
    ScalarType min_dist = 100.0;
    unsigned int best_candidate = 0;
    // first pass: choose nearest point
    for (unsigned int i = 0; i != m_mesh->m_vertices_number; i++)
    {
        p1 = m_mesh->m_current_positions.block_vector(i);
        
        ray_point_dist = ((p1-p0).cross(dir)).norm();
        if (ray_point_dist < min_dist)
        {
            min_dist = ray_point_dist;
            best_candidate = i;
        }
    }
    for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
    {
        AttachmentConstraint* ac;
        if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
        {
            ray_point_dist = ((ac->GetFixedPoint()-p0).cross(dir)).norm();
            if (ray_point_dist < min_dist)
            {
                min_dist = ray_point_dist;
                best_candidate = ac->GetConstrainedVertexIndex();
            }
        }
    }
    // exit if no one fits
    if (min_dist > DEFAULT_SELECTION_RADIUS)
    {
        return false;
    }
    // second pass: toggle that point's fixed position constraint
    bool current_state_on = false;
    for (std::vector<Constraint*>::iterator c = m_constraints.begin(); c != m_constraints.end(); ++c)
    {
        AttachmentConstraint* ac;
        if (ac = dynamic_cast<AttachmentConstraint*>(*c)) // is attachment constraint
        {
            if (ac->GetConstrainedVertexIndex() == best_candidate)
            {
                current_state_on = true;
                m_constraints.erase(c);
                break;
            }
        }
    }
    if (!current_state_on)
    {
        AddAttachmentConstraint(best_candidate);
    }

    return true;
}

void Simulation::SelectAtttachmentConstraint(AttachmentConstraint* ac)
{
    m_selected_attachment_constraint = ac;
    m_selected_attachment_constraint->Select();
}

void Simulation::UnselectAttachmentConstraint()
{
    if (m_selected_attachment_constraint)
    {
        m_selected_attachment_constraint->UnSelect();
    }
    m_selected_attachment_constraint = NULL;
}

void Simulation::AddAttachmentConstraint(unsigned int vertex_index)
{
    AttachmentConstraint* ac = new AttachmentConstraint(&m_stiffness_attachment, &m_stiffness_attachment_pbd, vertex_index, m_mesh->m_current_positions.block_vector(vertex_index));
    m_constraints.push_back(ac);
}

void Simulation::MoveSelectedAttachmentConstraintTo(const EigenVector3& target)
{
    if (m_selected_attachment_constraint)
        m_selected_attachment_constraint->SetFixedPoint(target);
}

void Simulation::clearConstraints()
{
    for (unsigned int i = 0; i < m_constraints.size(); ++i)
    {
        delete m_constraints[i];
    }
    m_constraints.clear();
}

void Simulation::copyDataToGPU(){
	GPUConstraint *Gconstraint=new GPUConstraint[m_constraints.size()];
	GPUPrimitive *Gprimitive=new GPUPrimitive[m_scene->m_primitives.size()];
	for(int i=0;i<m_constraints.size();++i){
		Gconstraint[i].stiffness=m_constraints[i]->Stiffness();
		Gconstraint[i].stiffnessPBD=m_constraints[i]->StiffnessPBD();
		Gconstraint[i].type=m_constraints[i]->type;
		if(Gconstraint[i].type==0){
			AttachmentConstraint *a=(AttachmentConstraint*)m_constraints[i];
			Gconstraint[i].fix_index=a->m_p0;
			EigenVector3 v=a->GetFixedPoint();
			Gconstraint[i].fixedPoint=glm::vec3(v.x(),v.y(),v.z());
		}
		else{
			SpringConstraint *s=(SpringConstraint*)m_constraints[i];
			Gconstraint[i].p1=s->m_p1;
			Gconstraint[i].p2=s->m_p2;
			Gconstraint[i].rest_length=s->m_rest_length;
			//cout<<Gconstraint[i].length<<endl;
		}
	}
	//copy constraint
	for(int i=0;i<m_scene->m_primitives.size();++i){
		Gprimitive[i].pos=m_scene->m_primitives[i]->m_pos;
		if(m_scene->m_primitives[i]->m_type==PLANE){
			Gprimitive[i].type=2;
			Plane *p=(Plane*)m_scene->m_primitives[i];
			Gprimitive[i].pNormal=p->m_normal;
		}
		else if(m_scene->m_primitives[i]->m_type==SPHERE){
			Gprimitive[i].type=1;
			Sphere *s=(Sphere*)m_scene->m_primitives[i];
			Gprimitive[i].radius=s->m_radius;
		}
		else if(m_scene->m_primitives[i]->m_type==CUBE){
			Gprimitive[i].type=0;
			Cube *c=(Cube*)m_scene->m_primitives[i];
			Gprimitive[i].cSize=c->m_hf_dims;
		}
	}
	//copy primitive
	int size=m_mesh->m_dim[0]*m_mesh->m_dim[1];
	glm::vec3 *pos=new glm::vec3[size];
	glm::vec3 *vel=new glm::vec3[size];
	for(int i=0;i<size;++i){
		EigenVector3 v=m_mesh->m_current_positions.block_vector(i);
		pos[i]=glm::vec3(v.x(),v.y(),v.z());
		v=m_mesh->m_current_velocities.block_vector(i);
		vel[i]=glm::vec3(v.x(),v.y(),v.z());
	}
	//copy position and velocity
	float mass=m_mesh->m_mass_matrix.coeff(0,0);
	copyData(Gconstraint,Gprimitive,pos,vel,m_mesh->m_dim[0],m_mesh->m_dim[1],m_constraints.size(),m_scene->m_primitives.size(),mass,m_restitution_coefficient);

	delete(pos);
	delete(vel);
}

void Simulation::setupConstraints()
{
	// LOOK setup spring constraints
    clearConstraints();

    switch(m_mesh->m_mesh_type)
    {
    case MESH_TYPE_CLOTH:
        // procedurally generate constraints including to attachment constraints
        {
            // generating attachment constraints.
            AddAttachmentConstraint(0);
            AddAttachmentConstraint(m_mesh->m_dim[1]*(m_mesh->m_dim[0]-1));

			// TODO
            // generate stretch constraints. assign a stretch constraint for each edge.
			EigenVector3 p1, p2;
            for(std::vector<Edge>::iterator e = m_mesh->m_edge_list.begin(); e != m_mesh->m_edge_list.end(); ++e)
            {
                p1 = m_mesh->m_current_positions.block_vector(e->m_v1);
                p2 = m_mesh->m_current_positions.block_vector(e->m_v2);
                SpringConstraint *c = new SpringConstraint(&m_stiffness_stretch, &m_stiffness_stretch_pbd, e->m_v1, e->m_v2, (p1-p2).norm());
                m_constraints.push_back(c);
            }

			
			// TODO
            // generate bending constraints. naive solution using cross springs 
			int X=m_mesh->m_dim[0];//width
			int Y=m_mesh->m_dim[1];//height
			for(int i=0;i<Y;i++){
				for(int j=0;j<X-2;j++){
					p1 = m_mesh->m_current_positions.block_vector(X*i+j);
					p2 = m_mesh->m_current_positions.block_vector(X*i+j+2);
					SpringConstraint *c=new SpringConstraint(&m_stiffness_bending,&m_stiffness_stretch_pbd,X*i+j,X*i+j+2,(p1-p2).norm());
					m_constraints.push_back(c);
				}
			}

			for(int i=0;i<X;i++){
				for(int j=0;j<Y-2;j++){
					p1 = m_mesh->m_current_positions.block_vector(Y*i+j);
					p2 = m_mesh->m_current_positions.block_vector(Y*i+j+2);
					SpringConstraint *c=new SpringConstraint(&m_stiffness_bending,&m_stiffness_stretch_pbd,Y*i+j,Y*i+j+2,(p1-p2).norm());
					m_constraints.push_back(c);
				}
			}

        }
        break;
    case MESH_TYPE_TET:
        {
            // generate stretch constraints. assign a stretch constraint for each edge.
            EigenVector3 p1, p2;
            for(std::vector<Edge>::iterator e = m_mesh->m_edge_list.begin(); e != m_mesh->m_edge_list.end(); ++e)
            {
                p1 = m_mesh->m_current_positions.block_vector(e->m_v1);
                p2 = m_mesh->m_current_positions.block_vector(e->m_v2);
                SpringConstraint *c = new SpringConstraint(&m_stiffness_stretch, &m_stiffness_stretch_pbd, e->m_v1, e->m_v2, (p1-p2).norm());
                m_constraints.push_back(c);
            }
        }
        break;
    }
	copyDataToGPU();
}

void Simulation::dampVelocity()
{
    if (std::abs(m_damping_coefficient) < EPSILON)
        return;

    // post-processing damping
    EigenVector3 pos_mc(0.0, 0.0, 0.0), vel_mc(0.0, 0.0, 0.0);
    unsigned int i, size;
    ScalarType denominator(0.0), mass(0.0);
    size = m_mesh->m_vertices_number;
    for(i = 0; i < size; ++i)
    {
        mass = m_mesh->m_mass_matrix.coeff(i*3, i*3);

        pos_mc += mass * m_mesh->m_current_positions.block_vector(i);
        vel_mc += mass * m_mesh->m_current_velocities.block_vector(i);
        denominator += mass;
    }
    assert(denominator != 0.0);
    pos_mc /= denominator;
    vel_mc /= denominator;

    EigenVector3 angular_momentum(0.0, 0.0, 0.0), r(0.0, 0.0, 0.0);
    EigenMatrix3 inertia, r_mat;
    inertia.setZero(); r_mat.setZero();

    for(i = 0; i < size; ++i)
    {
        mass = m_mesh->m_mass_matrix.coeff(i*3, i*3);

        r = m_mesh->m_current_positions.block_vector(i) - pos_mc;
        angular_momentum += r.cross(mass * m_mesh->m_current_velocities.block_vector(i));

        //r_mat = EigenMatrix3(0.0,  r.z, -r.y,
        //                    -r.z, 0.0,  r.x,
        //                    r.y, -r.x, 0.0);

        r_mat.coeffRef(0, 1) = r[2];
        r_mat.coeffRef(0, 2) = -r[1];
        r_mat.coeffRef(1, 0) = -r[2];
        r_mat.coeffRef(1, 2) = r[0];
        r_mat.coeffRef(2, 0) = r[1];
        r_mat.coeffRef(2, 1) = -r[0];

        inertia += r_mat * r_mat.transpose() * mass;
    }
    EigenVector3 angular_vel = inertia.inverse() * angular_momentum;

    EigenVector3 delta_v(0.0, 0.0, 0.0);
    for(i = 0; i < size; ++i)
    {
        r = m_mesh->m_current_positions.block_vector(i) - pos_mc;
        delta_v = vel_mc + angular_vel.cross(r) - m_mesh->m_current_velocities.block_vector(i);     
        m_mesh->m_current_velocities.block_vector(i) += m_damping_coefficient * delta_v;
    }
}

void Simulation::calculateExternalForce()
{
	VectorX external_acceleration_field(m_mesh->m_system_dimension);
	external_acceleration_field.setZero();

    // LOOK: adding gravity
    for (unsigned int i = 0; i < m_mesh->m_vertices_number; ++i)
    {
        external_acceleration_field[3*i+1] += -m_gravity_constant;
    }

    m_external_force = m_mesh->m_mass_matrix * external_acceleration_field;
}

void Simulation::detectCollision(const VectorX& x, std::vector<CollisionInstance>& collisions)
{
    // Naive implementation of collision detection
    EigenVector3 normal;
    ScalarType dist;
	collisions.clear();

    for (unsigned int i = 0; i < m_mesh->m_vertices_number; ++i)
    {
        EigenVector3 xi = x.block_vector(i);

        if (m_scene->StaticIntersectionTest(xi, normal, dist)) 
        {
			// if collision
			collisions.push_back(CollisionInstance(i, normal, dist));
        }
    }
}

/*void Simulation::detectCollision(const VectorX& x,std::vector<CollisionInstance>& collisions)
{
    // Naive implementation of collision detection
	collisions.clear();
    
	glm::vec3 *xi=new glm::vec3[m_mesh->m_vertices_number];
	glm::vec3 *normal=new glm::vec3[m_mesh->m_vertices_number];
	float *dist=new float[m_mesh->m_vertices_number];
	for(int i=0;i<m_mesh->m_vertices_number;++i){
		EigenVector3 tmp = x.block_vector(i);
		xi[i]=glm::vec3(tmp.x(),tmp.y(),tmp.z());
	}
	
	bool* result=collisionDetection(xi,normal,dist,m_mesh->m_vertices_number);
	for(unsigned int i=0;i<m_mesh->m_vertices_number;++i){
		if(result[i]){ 
			EigenVector3 n;
			n.x()=normal[i].x;n.y()=normal[i].y;n.z()=normal[i].z;
			collisions.push_back(CollisionInstance(i,n,dist[i]));
		}
	}
	delete(xi);
	delete(result);
}*/

void Simulation::resolveCollision(VectorX&x, VectorX& v, std::vector<CollisionInstance>& collisions)
{
	for (std::vector<CollisionInstance>::iterator it = collisions.begin(); it!= collisions.end(); ++it)
	{
		// find the vertex id
		unsigned int id = it->point_index;
		// correct position
		x.block_vector(id) -= it->normal * it->dist;
		// correct velocity
		ScalarType vn = it->normal.dot(v.block_vector(id));
		v.block_vector(id) += -(1+m_restitution_coefficient)*vn*it->normal;
	}
}

void Simulation::integrateExplicitEuler(VectorX& x, VectorX& v, ScalarType dt)
{
	// TODO:
	// v_next = v_current + dt * a_current
	// x_next = x_current + dt * v_current
	//VectorX temp_x, temp_v; // temp_x and temp_v are used to compute the forces in an intermediate state

	// step 1: compute k1
	VectorX f1;
	computeForces(x, f1);
	// get the slope for x and v separately 
	VectorX k1_x = v;
	VectorX k1_v = m_mesh->m_inv_mass_matrix * f1;
	// setup a temporary point for next iteration
	//temp_x = x + half_dt * k1_x;
	//temp_v = v + half_dt * k1_v;

	//x = x + 1/6.0 * dt * (k1_x + 2*k2_x + 2*k3_x + k4_x);
	//v = v + 1/6.0 * dt * (k1_v + 2*k2_v + 2*k3_v + k4_v);
	x=x+dt*k1_x;
	v=v+dt*k1_v;
}

void Simulation::integrateExplicitRK2(VectorX& x, VectorX& v, ScalarType dt)
{
	// TODO:
	// heun's method (RK2)
	VectorX temp_x, temp_v; // temp_x and temp_v are used to compute the forces in an intermediate state

	// step 1: compute k1
	VectorX f1;
	computeForces(x, f1);
	// get the slope for x and v separately 
	VectorX k1_x = v;
	VectorX k1_v = m_mesh->m_inv_mass_matrix * f1;
	// setup a temporary point for next iteration
	temp_x = x + dt * k1_x;
	temp_v = v + dt * k1_v;

	// step 2: compute k2
	VectorX f2;
	computeForces(temp_x, f2);
	// get the slope for x and v separately 
	VectorX k2_x = temp_v;
	VectorX k2_v = m_mesh->m_inv_mass_matrix * f2;
	// setup a temporary point for next iteration
	//temp_x = x + dt * k1_x;
	//temp_v = v + dt * k1_v;

	//x = x + 1/6.0 * dt * (k1_x + 2*k2_x + 2*k3_x + k4_x);
	//v = v + 1/6.0 * dt * (k1_v + 2*k2_v + 2*k3_v + k4_v);
	x=x+dt*k2_x;
	v=v+dt*k2_v;
}

void Simulation::integrateExplicitRK4(VectorX& x, VectorX& v, ScalarType dt)
{
	// LOOK: here's a sample work flow of RK4
	ScalarType half_dt = dt / 2.0;
	VectorX temp_x, temp_v; // temp_x and temp_v are used to compute the forces in an intermediate state

	// step 1: compute k1
	VectorX f1;
	computeForces(x, f1);
	// get the slope for x and v separately 
	VectorX k1_x = v;
	VectorX k1_v = m_mesh->m_inv_mass_matrix * f1;
	// setup a temporary point for next iteration
	temp_x = x + half_dt * k1_x;
	temp_v = v + half_dt * k1_v;

	// step2: compute k2
	VectorX f2;
	computeForces(temp_x, f2);
	// get the slope for x and v separately 
	VectorX k2_x = temp_v;
	VectorX k2_v = m_mesh->m_inv_mass_matrix * f2;
	// setup a temporary point for next iteration
	temp_x = x + half_dt * k2_x;
	temp_v = v + half_dt * k2_v;

	// step3: compute k3
	VectorX f3;
	computeForces(temp_x, f3);
	// get the slope for x and v separately 
	VectorX k3_x = temp_v;
	VectorX k3_v = m_mesh->m_inv_mass_matrix * f3;
	// setup a temporary point for next iteration
	temp_x = x + dt * k2_x;
	temp_v = v + dt * k2_v;

	// step4: compute k4
	VectorX f4;
	computeForces(temp_x, f4);
	// get the slope for x and v separately 
	VectorX k4_x = temp_v;
	VectorX k4_v = m_mesh->m_inv_mass_matrix * f4;

	// Put it all together
	x = x + 1/6.0 * dt * (k1_x + 2*k2_x + 2*k3_x + k4_x);
	v = v + 1/6.0 * dt * (k1_v + 2*k2_v + 2*k3_v + k4_v);
}

void Simulation::integrateExplicitSymplectic(VectorX& x, VectorX& v, ScalarType dt)
{
	// TODO:
	VectorX f1;
	computeForces(x, f1);
	// get the slope for x and v separately 
	VectorX k1_x = v;
	VectorX k1_v = m_mesh->m_inv_mass_matrix * f1;

	//x=x+dt*k1_x;
	v=v+dt*k1_v;

	k1_x=v;
	//k1_v=m_mesh->m_inv_mass_matrix * f1;

	x=x+dt*k1_x;
}

void Simulation::integrateImplicitBW(VectorX& x, VectorX& v, ScalarType dt)
{
	// TODO:
	// v_next = v_current + dt * a_next
	// x_next = x_current + dt * v_next
	// [M - dt^2 K] * v_next = M * v_current + dt * f(x_current);
	// A * v_next = b;
}

void Simulation::integratePBD(VectorX& x, VectorX& v, unsigned int ns)
{
	// TODO:
	/*v = v + m_h*m_mesh->m_inv_mass_matrix*m_external_force;
	//tmp update the p
	VectorX p = x;
	p = x + m_h*v;
	
	//project the p
	for (int k = 0; k<ns; k++)
	{
		if (glm::mod(float(k),2.f)==0)
		{
			for (int i = 0; i<m_constraints.size() ;i++)
			{
				m_constraints[i]->PBDProject(p, m_mesh->m_inv_mass_matrix, k);
			}
		}
		else
		{
			for (int i = 0; i<m_constraints.size() ;i++)
			{
				m_constraints[i]->PBDProject(p, m_mesh->m_inv_mass_matrix, k);
			}
		}
		
	}

	//update the v and x
	v = (p - x)/m_h;
	x = p;*/
	float dt = m_h / m_iterations_per_frame;
	float dm=m_mesh->m_total_mass/(m_mesh->m_dim[0]*m_mesh->m_dim[1]);
	//float dm=m_mesh->m_total_mass/ns;
	VectorX f,p;
	//computeForces(x,f);
	calculateExternalForce();
	f=this->m_external_force;
	v=v+dt*f/dm;
	//v = v + m_h * m_mesh->m_inv_mass_matrix * f;
	p=x+dt*v;

	for(std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
    {
		(*it)->PBDProject(p,m_mesh->m_inv_mass_matrix,ns);
	}
	//SpringConstraint *c = new SpringConstraint(&m_stiffness_stretch, &m_stiffness_stretch_pbd, e->m_v1, e->m_v2, (p1-p2).norm());
   // m_constraints.push_back(c);

	v=(p-x)/dt;
	x=p;
}

#pragma region implicit/explicit euler
void Simulation::computeForces(const VectorX& x, VectorX& force)
{
    VectorX gradient;

    gradient.resize(m_mesh->m_system_dimension);
    gradient.setZero();

    // springs
    for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
    {
        (*it)->EvaluateGradient(x, gradient);
    }

	// internal_force = - gradient of elastic energy
    force = -gradient;

	// external forces
	force += m_external_force;
}

void Simulation::computeStiffnessMatrix(const VectorX& x, SparseMatrix& stiffness_matrix)
{
	SparseMatrix hessian;
	hessian.resize(m_mesh->m_system_dimension, m_mesh->m_system_dimension);
	std::vector<SparseMatrixTriplet> h_triplets;
	h_triplets.clear();

	for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
	{
		(*it)->EvaluateHessian(x, h_triplets);
	}

	hessian.setFromTriplets(h_triplets.begin(), h_triplets.end());

	// stiffness_matrix = - hessian matrix
	stiffness_matrix = - hessian;
}

#pragma endregion

#pragma region utilities
void Simulation::factorizeDirectSolverLLT(const SparseMatrix& A, Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>& lltSolver, char* warning_msg)
{
    SparseMatrix A_prime = A;
    lltSolver.analyzePattern(A_prime);
    lltSolver.factorize(A_prime);
    ScalarType Regularization = 0.00001;
    bool success = true;
    while (lltSolver.info() != Eigen::Success)
    {
        Regularization *= 10;
        A_prime = A_prime + Regularization*m_mesh->m_identity_matrix;
        lltSolver.factorize(A_prime);
        success = false;
    }
    if (!success)
        std::cout << "Warning: " << warning_msg <<  " adding "<< Regularization <<" identites.(llt solver)" << std::endl;
}

void Simulation::factorizeDirectSolverLDLT(const SparseMatrix& A, Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>& ldltSolver, char* warning_msg)
{
    SparseMatrix A_prime = A;
    ldltSolver.analyzePattern(A_prime);
    ldltSolver.factorize(A_prime);
    ScalarType Regularization = 0.00001;
    bool success = true;
    while (ldltSolver.info() != Eigen::Success)
    {
        Regularization *= 10;
        A_prime = A_prime + Regularization*m_mesh->m_identity_matrix;
        ldltSolver.factorize(A_prime);
        success = false;
    }
    if (!success)
        std::cout << "Warning: " << warning_msg <<  " adding "<< Regularization <<" identites.(ldlt solver)" << std::endl;
}

void Simulation::generateRandomVector(const unsigned int size, VectorX& x)
{
    x.resize(size);

    for (unsigned int i = 0; i < size; i++)
    {
        x(i) = ((ScalarType)(rand())/(ScalarType)(RAND_MAX+1)-0.5)*2;
    }
}

void Simulation::computeSystemMatrix()
{
	SparseMatrix M = m_mesh->m_mass_matrix;
	SparseMatrix K;
	VectorX x;
	computeStiffnessMatrix(x, K);
	ScalarType dt = m_h / m_iterations_per_frame;
	// fill A
	m_A = M - dt*dt*K;

	vector<int> host_Rows;
	vector<int> host_Cols;
	vector<float> host_Val;
	// copy to GPU

	for (int k = 0; k < m_A.outerSize(); ++k)
	{
		for (SparseMatrix::InnerIterator it(m_A, k); it; ++it)
		{
			host_Val.push_back(it.value());
			host_Rows.push_back(it.row());   // row index
			host_Cols.push_back(it.col());   // col index (here it is equal to k)
		}
	}

	convertSystemMatrix(host_Rows, host_Cols, host_Val);
	
		
}

#pragma endregion
