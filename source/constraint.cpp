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

#include "constraint.h"

//----------Constraint Class----------//
Constraint::Constraint(ScalarType *stiffness, ScalarType *pbd_stiffness) : 
    m_p_stiffness(stiffness),
	m_p_pbd_stiffness(pbd_stiffness)
{
}

Constraint::Constraint(const Constraint& other) : 
    m_p_stiffness(other.m_p_stiffness),
	m_p_pbd_stiffness(other.m_p_pbd_stiffness)
{
}

Constraint::~Constraint()
{
}

//----------AttachmentConstraint Class----------//
AttachmentConstraint::AttachmentConstraint(ScalarType *stiffness, ScalarType *pbd_stiffness) : 
    Constraint(stiffness, pbd_stiffness)
{
    m_selected = false;
	type=0;
}

AttachmentConstraint::AttachmentConstraint(ScalarType *stiffness, ScalarType *pbd_stiffness, unsigned int p0, const EigenVector3& fixedpoint) : 
    Constraint(stiffness, pbd_stiffness),
    m_p0(p0),
    m_fixd_point(fixedpoint)
{
    m_selected = false;
	type=0;
}

AttachmentConstraint::AttachmentConstraint(const AttachmentConstraint& other) : 
    Constraint(other),
    m_p0(other.m_p0),
    m_fixd_point(other.m_fixd_point),
    m_selected(other.m_selected)
{
    type=0;
}

AttachmentConstraint::~AttachmentConstraint()
{
}

// attachment spring gradient: k*(current_length)*current_direction
void AttachmentConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	// LOOK
    EigenVector3 g_i = (*(m_p_stiffness))*(x.block_vector(m_p0) - m_fixd_point);
    gradient.block_vector(m_p0) += g_i;
}

void AttachmentConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets)
{
	// LOOK
    ScalarType ks = *(m_p_stiffness);
    hessian_triplets.push_back(SparseMatrixTriplet(3*m_p0, 3*m_p0, ks));
    hessian_triplets.push_back(SparseMatrixTriplet(3*m_p0+1, 3*m_p0+1, ks));
    hessian_triplets.push_back(SparseMatrixTriplet(3*m_p0+2, 3*m_p0+2, ks));
}

void AttachmentConstraint::PBDProject(VectorX& x, const SparseMatrix& inv_mass, unsigned int ns)
{
	// LOOK
	ScalarType k_prime = 1 - std::pow(1-*(m_p_pbd_stiffness), 1.0/ns);
	
	EigenVector3 p = x.block_vector(m_p0);
	EigenVector3 dp = m_fixd_point-p;

	x.block_vector(m_p0) += k_prime * dp;
}

void AttachmentConstraint::Draw(const VBO& vbos)
{
    m_attachment_constraint_body.move_to(Eigen2GLM(m_fixd_point));
    if (m_selected)
        m_attachment_constraint_body.change_color(glm::vec3(0.8, 0.8, 0.2));
    else
        m_attachment_constraint_body.change_color(glm::vec3(0.8, 0.2, 0.2));
        
    m_attachment_constraint_body.Draw(vbos);
}

//----------SpringConstraint Class----------//
SpringConstraint::SpringConstraint(ScalarType *stiffness, ScalarType *pbd_stiffness) : 
    Constraint(stiffness, pbd_stiffness)
{
	type=1;
}

SpringConstraint::SpringConstraint(ScalarType *stiffness, ScalarType *pbd_stiffness, unsigned int p1, unsigned int p2, ScalarType length) : 
    Constraint(stiffness, pbd_stiffness),
    m_p1(p1),
    m_p2(p2),
    m_rest_length(length)
{
	type=1;
}

SpringConstraint::SpringConstraint(const SpringConstraint& other) : 
    Constraint(other),
    m_p1(other.m_p1),
    m_p2(other.m_p2),
    m_rest_length(other.m_rest_length)
{
	type=1;
}

SpringConstraint::~SpringConstraint()
{
}

// sping gradient: k*(current_length-rest_length)*current_direction;
void SpringConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	// TODO
	//EigenVector3 g_i = (*(m_p_stiffness))*(x.block_vector(m_p0) - m_fixd_point);
    //gradient.block_vector(m_p0) += g_i;

	EigenVector3 p1=x.block_vector(m_p1);
	EigenVector3 p2=x.block_vector(m_p2);
	float currentLength=(p1-p2).norm();
	float force=(*this->m_p_stiffness)*(currentLength-this->m_rest_length);
	EigenVector3 n1=p1-p2,n2=p2-p1;
	n1.normalize();n2.normalize();
	gradient.block_vector(m_p1)+=n1*force;
	gradient.block_vector(m_p2)+=n2*force;
}

void SpringConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets)
{
	// TODO
}

void SpringConstraint::PBDProject(VectorX& x, const SparseMatrix& inv_mass, unsigned int ns)
{
	// TODO
	// change project order
	ScalarType k_prime = 1 - std::pow(1-*(m_p_pbd_stiffness), 1.0/ns);
	float rest_length=m_rest_length;
	//cout<<m_rest_length<<endl;
	EigenVector3 p1 = x.block_vector(m_p1);
	EigenVector3 p2 = x.block_vector(m_p2);

	float current_length=(p1-p2).norm();
	EigenVector3 current_direction=(p1-p2)/current_length;
	EigenVector3 dp=(current_length-rest_length)*current_direction;

	ScalarType w1=inv_mass.coeff(m_p1,m_p1);
	ScalarType w2=inv_mass.coeff(m_p2,m_p2);
	
	x.block_vector(m_p1) -= k_prime * dp*w1/(w1+w2);
	x.block_vector(m_p2) += k_prime * dp*w2/(w1+w2);
}