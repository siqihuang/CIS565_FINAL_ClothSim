#ifndef _KDTREE_H_
#define _KDTREE_H_
#include <iostream>
#include <glm.hpp>
#include <vector>
#include <algorithm>
using namespace std;

class kdtree{
public:
	kdtree *lc,*rc;
	int depth;
	int index;
	glm::vec3 position;
	vector<float*> sortVector;
	vector<unsigned int>lcTri,rcTri,Triangles;
	vector<unsigned short>*indices;
	vector<glm::vec3> *positions;
	float xMax,yMax,zMax,xMin,yMin,zMin,xCenter,yCenter,zCenter;

	kdtree();
	void createTree(int depth,vector<glm::vec3> *positions,vector<unsigned short> *indices,vector<unsigned int> Triangles);
	void findBoundary();
	void setPosition(glm::vec3 position);
	float findTriangleBoundary(int n,int axis,int type);

	bool insideBox(glm::vec3 pos);
	void getNearbyTriangles(glm::vec3 pos,vector<unsigned short> &list);
};

#endif