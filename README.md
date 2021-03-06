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

Presentation Slides Link: (https://docs.google.com/presentation/d/11hLZRSBLbAv0bsrRw1vXXsuu3ncNj1UpvBmuZB7ff5E/edit?usp=sharing)

### MileStone2 (11/30/2015)

Progress:

1. finish the implementation of implicit method using cuSolver
2. finish the AABB tree collision detection and resolve using imported obj file
3. finish damp velocity computation on GPU

Next Step:

1. implement the tearing-apart effect
2. optimize the collision detection
3. detailed performance analysis
4. improving the shader

Presentation Slides Link: (https://docs.google.com/presentation/d/1w5eOXz4IK8DZFQqKDkIC5HjKT8hTvjFJlC4LQjbyFWg/edit?usp=sharing)

### MileStone3 (12/7/2015)

Progress:

1. finish the implementation of cloth trearing feature
2. finish the implementation of cloth self-collision
3. finish the implementation of new shader


Next Step:

1. Performance Analysis
2. Demo Creation
3. "*** Polish the code and adding new features (user interaction, etc.) ***"

Presentation Slides Link: (https://docs.google.com/presentation/d/17yJgBsn3dMI6hz9sOIewUYsKLIjcYYmOaqrHFI2k5ks/edit?usp=sharing)

### Final Presentation (12/11/2015)

Slides Link: (https://docs.google.com/presentation/d/1ywdBxPHv_mPhfjz4zERtnLXnV_xNAmKroLEGFNd2UQQ/edit?usp=sharing)

Algorithm
========================

We are using two way of integration method for the simulation of cloth mesh. One is force based method, the other is position based method.

### Force Based Method

In the force based method, we treat each point on the cloth mesh as a mass point, which has the property of position and velocity. We attach spring for each pair of neighbour points and apply Hook's Law to calculate the internal force and update the position and velocity each frame.

![](https://github.com/siqihuang/CIS565_FINAL_ClothSim/blob/master/pic/force_based_method.png?raw=true)

We test Explicit Method (RK1, Rk2, RK4) and Implicit Method seperately and compare their stability and computation time. We found that for the explicit method, the higher order the RK method is, the more stable the simulation is (We are using the largest simulation tim step to measure this part). The implicit mehthod is way much more stable than any of the explicit method, but it requires much more computaion time. Also, we have compared the performance of CPU version and GPU version of the simulation (more detail on latter Performance Analysis section). However, we found it strange that the GPU version of implicit method implemented using the cuSolver is slower than the CPU verison implemented by the Eigen Solver. We suspect that this is caused by the I/O bottleneck in GPU (since we need to formulize the data to feed into the cuda solver) or the Eigen is super optimized in solving this kind of linear system.

### Position Based Method

This part we are implementing the algorithm given by Position Base Dynamics (http://matthias-mueller-fischer.ch/publications/posBasedDyn.pdf) The main workflow is as follows:

![](https://github.com/siqihuang/CIS565_FINAL_ClothSim/blob/master/pic/pbd_workflow.png?raw=true)

Performance Analysis
========================
The following time is in millisecond
#1 Different Method: PBD VS RK4
collision with cube:
========================
![](image/pbd-cube-share.png)
![](image/rk4-cube-share.png)

![](image/pbd-cube-value.png)
![](image/pbd-cube-value.png)

collision with obj:
========================
![](image/pbd-obj-share.png)
![](image/rk4-obj-share.png)

![](image/pbd-obj-value.png)
![](image/pbd-obj-value.png)

time used in different process:
========================
![](image/pbd-mul-obj-value.png)
![](image/rk4-mul-obj-value.png)

It is obvious that in both method, the integration in CPU takes most of the time and share. As PBD is much faster than RK4, it uses less time in both CPU and GPU

#2 Different Grid-Size:
grid size 50(cube vs obj):
========================
![](image/cube-50-share.png)
![](image/obj-50-share.png)

![](image/cube-50-value.png)
![](image/obj-50-value.png)

grid size 100(cube vs obj):
========================
![](image/cube-100-share.png)
![](image/obj-100-share.png)

![](image/cube-100-value.png)
![](image/obj-100-value.png)

grid size 200(cube vs obj):
========================
![](image/cube-200-share.png)
![](image/obj-200-share.png)

![](image/cube-200-value.png)
![](image/obj-200-value.png)

time used between different process(cube vs obj)
========================
![](image/cube-mul-size.png)
![](image/obj-mul-size.png)

We can see clearly that as the mesh size goes up, for the GPU part both the integration and collision goes up. But the collision in obj collision case goes up aggressively. This indicates that the collision with obj uses most of the time when the mesh size is large. That is very easy to understand because the actual collision detection process is not in a parallel way, which makes it similar with CPU. For the cube intersection case, it is a very parallel process. So the time is relatively short here.

#3 bottleneck:
GPU bottleneck
========================
![](image/GPU-collision.png)
![](image/CPU-collision.png)

the left image is the collision with object in GPU, and the right one is in CPU. We can see the collision time used in GPU is even higher in CPU when colliding with object. And the trend of the time increase as the object size increase in GPU is similar in CPU, which indicate that the GPU side also suffer from the problem of collision detection. So the bottleneck in GPU is in collision detection with object.

CPU bottleneck
========================
![](image/GPU-integration.png)
![](image/CPU-integration.png)

the left image is the integration time in GPU while the right one is the time in CPU. In either case CPU uses more time in integration, which makes it the bottleneck for CPU.

Demo Video
========================
[![video1](http://img.youtube.com/vi/uJImY-I7LBU/0.jpg)](http://www.youtube.com/watch?v=uJImY-I7LBU)
[![video2](http://img.youtube.com/vi/bkrS-DjJZrM/0.jpg)](http://www.youtube.com/watch?v=bkrS-DjJZrM)
[![video3](http://img.youtube.com/vi/jIAqLqZhTg0/0.jpg)](http://www.youtube.com/watch?v=jIAqLqZhTg0)
[![video4](http://img.youtube.com/vi/n8ZgMlkArFg/0.jpg)](http://www.youtube.com/watch?v=n8ZgMlkArFg)
[![video5](http://img.youtube.com/vi/tuhjUOxhpA0/0.jpg)](http://www.youtube.com/watch?v=tuhjUOxhpA0)
