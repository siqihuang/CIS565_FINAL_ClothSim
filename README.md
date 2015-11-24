GPU-Based Cloth Simulation
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Ziye Zhou & Siqi Huang

Progress
========================
### Project Overview (11/16/2015)

Motivation for Project:

Cloth simulation is an importpart branch in the physical based simulation in Computer Graphics. People are using this technique to generate tons of cool effect in games, movies, etc. Lots of research work has been done to implement and refine the simulation algorithm ,but most of them are restricted on CPU (see the reference). This give us the motivation to extend some of the state of art algorithm to GPU version to further speed up the simulation. A good start is that we already have a CPU version of the cloth simulation(the rendered video is in attached file) implemented in CIS 563. From that project, we noticed that for small object with a relative low resolution, combined with small number of iterations, we can achieve real time simulation with acceptable artifacts. However, If the number of objects is large (say we have a large scene to simualte) and the mesh is of high resolution, or we want really accurate detail which requires large number of iterations to converge, the CPU bottlenect is reached and it is no longer possible to render in real time. Having requirement like these and given the special parallel characterastic of simulation algorithm, it is very suitable for the GPU to show its strength over the CPU.

What to do in the Project:

Since we already have a CPU version of the simulation, more work should be done in the acceleration and extra feature of the cloth simulation. In general we do it in the following n ways:
Implement the basic GPU version of the cloth simulation(transfer CPU algorithm to GPU)
Compare the performance analysis of the CPU and GPU version, find the bottleneck of each and try to refine the GPU version
Implement extra feature and deploy new scene(only on scene in CPU version) and also compare performance

### MileStone1 (11/23/2015)
Progress:

1. finish implementing force based integration method (explicit euler, RK2, RK4) 
2. finish implementing position based integration method
3. finish implementing simple primitive collision detection and resolve method

Next Step:

1. implicit method (require using cuSolver to solve linear system with sparse matrix)
2. AABB tree collision detection and resolve using imported obj file
3. more advanced integration method (projective dynamics)
4. new simulation effect (tear apart, etc)

Presentation Slides Link: (https://docs.google.com/presentation/d/11hLZRSBLbAv0bsrRw1vXXsuu3ncNj1UpvBmuZB7ff5E/edit)

