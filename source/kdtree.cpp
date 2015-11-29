#include "kdtree.h"

bool cmp(float *p,float *q)
{
	return p[0]<q[0];
}

kdtree::kdtree(){
	index=-1;
}

kdtree::kdtree(kdtree *root){
	this->depth=root->depth;
	this->index=root->index;
	this->xMax=root->xMax;
	this->xMin=root->xMin;
	this->yMax=root->yMax;
	this->yMin=root->yMin;
	this->zMax=root->zMax;
	this->zMin=root->zMin;
	this->sortVector=root->sortVector;
	this->Triangles=root->Triangles;
	this->indices=root->indices;
	this->positions=root->positions;
	this->lc=nullptr;
	this->rc=nullptr;
}

void kdtree::createTree(int depth,vector<glm::vec3> *positions,vector<unsigned short> *indices,vector<unsigned int> Triangles){
	this->depth=depth;
	this->Triangles=Triangles;
	this->positions=positions;
	this->indices=indices;
	lc=nullptr;
	rc=nullptr;

	findBoundary();

	if(Triangles.size()==1){		
		index=Triangles[0];
		//cout<<xMax<<","<<xMin<<endl;
		//cout<<yMax<<","<<yMin<<endl;
		//cout<<zMax<<","<<zMin<<endl;
		return;
	}

	lc=new kdtree();
	lc->setPosition(position);
	lc->createTree(depth+1,positions,indices,this->lcTri);
	rc=new kdtree();
	rc->setPosition(position);
	rc->createTree(depth+1,positions,indices,this->rcTri);
}

float kdtree::findTriangleBoundary(int n,int axis,int type){
	unsigned int index,index1,index2,index3;
	float value1,value2,value3,min,max;
	index=Triangles[n];
	index1=(*indices)[3*index];
	index2=(*indices)[3*index+1];
	index3=(*indices)[3*index+2];
	if(axis==0){//x
		value1=(*positions)[index1].x;
		value2=(*positions)[index2].x;
		value3=(*positions)[index3].x;
	}
	else if(axis==1){//y
		value1=(*positions)[index1].y;
		value2=(*positions)[index2].y;
		value3=(*positions)[index3].y;
	}
	else{//z
		value1=(*positions)[index1].z;
		value2=(*positions)[index2].z;
		value3=(*positions)[index3].z;
	}
	min=value1<value2?value1:value2;
	min=min<value3?min:value3;
	max=value1>value2?value1:value2;
	max=max>value3?max:value3;
	if(type==0)
		return min;
	else 
		return max;
}

void kdtree::findBoundary(){
	int size=Triangles.size();

	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,0,0);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	xMin=sortVector[0][0]+position.x;
	//xMin
	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,0,1);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	xMax=sortVector[size-1][0]+position.x;
	//xMax
	if(depth%3==0){
		for(int i=0;i<size/2;i++)
			lcTri.push_back(sortVector[i][1]);
		for(int i=size/2;i<size;i++)
			rcTri.push_back(sortVector[i][1]);
	}
	
	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,1,0);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	yMin=sortVector[0][0]+position.y;
	//yMin
	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,1,1);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	yMax=sortVector[size-1][0]+position.y;
	//yMax
	if(depth%3==1){
		for(int i=0;i<size/2;i++)
			lcTri.push_back(sortVector[i][1]);
		for(int i=size/2;i<size;i++)
			rcTri.push_back(sortVector[i][1]);
	}

	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,2,0);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	zMin=sortVector[0][0]+position.z;
	//zMin
	sortVector.clear();
	for(int i=0;i<size;i++){
		float *tmp=new float[2];
		tmp[0]=findTriangleBoundary(i,2,1);
		tmp[1]=Triangles[i];
		sortVector.push_back(tmp);
	}
	std::sort(sortVector.begin(),sortVector.end(),cmp);
	zMax=sortVector[size-1][0]+position.z;
	//zMax
	if(depth%3==2){
		for(int i=0;i<size/2;i++)
			lcTri.push_back(sortVector[i][1]);
		for(int i=size/2;i<size;i++)
			rcTri.push_back(sortVector[i][1]);
	}

	xCenter=(xMax+xMin)/2;yCenter=(yMax+yMin)/2;zCenter=(zMax+zMin)/2;
	float xhalf=xMax-xCenter,yhalf=yMax-yCenter,zhalf=zMax-zCenter;
	if(xhalf<0.05) xhalf=0.05;
	if(yhalf<0.05) yhalf=0.05;
	if(zhalf<0.05) zhalf=0.05;
	int n=1;
	//cout<<xMax<<","<<xMin<<endl;
	xMax=xCenter+xhalf*n;xMin=xCenter-xhalf*n;
	yMax=yCenter+yhalf*n;yMin=yCenter-yhalf*n;
	zMax=zCenter+zhalf*n;zMin=zCenter-zhalf*n;
	//cout<<xMax<<","<<xMin<<endl;
	//getchar();
	/*cout<<lcTri.size()<<endl;
	cout<<rcTri.size()<<endl;
	getchar();*/
}

void kdtree::setPosition(glm::vec3 position){
	this->position=position;
}

bool kdtree::insideBox(glm::vec3 pos){
	//if(index!=-1) cout<<index<<endl;
	if(pos.x<=xMax&&pos.x>=xMin&&pos.y<=yMax&&pos.y>=yMin&&pos.z<=zMax&&pos.z>=zMin){
		/*cout<<"test"<<endl;
		cout<<xMax<<","<<xMin<<endl;
		cout<<yMax<<","<<yMin<<endl;
		cout<<zMax<<","<<zMin<<endl;
		cout<<pos.x<<","<<pos.y<<","<<pos.z<<endl;
		cout<<"test end"<<endl;*/
		return true;
	}
	else return false;
}

void kdtree::getNearbyTriangles(glm::vec3 pos,vector<unsigned short> &list){
	//if(index!=-1) cout<<index<<endl;
	//cout<<index<<endl;
	if(!insideBox(pos)) return;
	if(index!=-1) list.push_back(index);
	else{
		lc->getNearbyTriangles(pos,list);
		rc->getNearbyTriangles(pos,list);
	}
}