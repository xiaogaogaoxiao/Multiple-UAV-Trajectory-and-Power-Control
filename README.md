## Multiple UAV Trajectory and Power Control

![](https://github.com/Vito-Swift/Multiple-UAV-Trajectory-and-Power-Control/raw/master/doc/page.png)

Simulations and documentations of a multiple UAV trajectory and power control algorithm based on SCA algorithm. This TPC algorithm is proposed by an undergraduate student group in CUHK(SZ) and submitted as a course project in EIE3280, 2019 summer.

#### Repository Structure

```bash
Root
|
+-- data
|    |
|    +-- trajectory*_iteration*.csv  % simulation results
+-- doc
|    |
|    +-- pre                         % presenation materials (beamer, etc.)
|    |
|    +-- report                      % report materials 
+-- result
|    |
|    +-- init_rate.png               % plotting of the initial transmission rate
|    |
|    +-- init_trajectory.png         % plotting of the initial trajectory
|    |
|    +-- opt_rate.png                % plotting of the optimized transmission rate
|    |
|    +-- opt_trajectory.png          % plotting of the optimized trajectory
+-- simulation
|    |
|	   +-- init                        % automated scripts to generate initial trajectories
|    |
|    +-- *.py                        % primary simulation scripts
```

#### Python Package Requirements

- numpy
- matplotlib
- cvxpy

#### Theoretical Analysis

You can see our theoretical analysis in [docs](https://github.com/Vito-Swift/EIE3280-CourseProj-TPC/tree/master/doc) and one can also refer to a [paper](https://arxiv.org/pdf/1809.05697.pdf) on optimization research by C. Shen, T.H. Chang, et al.

#### Simulation Result

- Simulation Scale
  - Number of UAV-BS pairs: K = 4
  - Number of time slots: M = 160
- Parameters
  - d_min = 20
  - v_l = 20, v_a = v_d = 5
  - h_min = 100, h_max = 200
  - P_max = 30
  - Communication bandwidth: B = 10 MHz
  - Power spectral density of addictive white Gaussian noise: N_0 = -160 dbm/Hz
- Initial locations
  - UAV: (0, 0, 100), (30, 0, 100), (0, 30, 100), (30, 30, 100)
  - Base station: (300, 0, 0), (100, 600, 0), (700, 700, 0), (100, 800, 0)

##### Initial Setup

***Trajectory:***

![](https://raw.githubusercontent.com/Vito-Swift/EIE3280-CourseProj-TPC/master/result/init_trajectory.png)

***Achievable Rate:***

![](https://raw.githubusercontent.com/Vito-Swift/EIE3280-CourseProj-TPC/master/result/init_rate.png)

##### Optimized Result

***Trajectory:***

![](https://raw.githubusercontent.com/Vito-Swift/EIE3280-CourseProj-TPC/master/result/opt_trajectory.png)

***Achievable Rate:***

![](https://raw.githubusercontent.com/Vito-Swift/EIE3280-CourseProj-TPC/master/result/opt_rate.png)

#### Remark

1. Based on the simulation one can see that the aggregate achievable transmission rate has been raised during the time slots.
2. The problem is formulated as a non-convex problem, thus the simulation will be dramatically different if one changes the initial trajectory. The result shows only a local minimum based on the initial trajectories.
3. Until now we did not give our solutions to calculate minimum time slot M yet, but keep this in mind, due to the symmetric and heuristic analysis, M should be much smaller than the total permitted time. (10 - 2 x smaller)
4. The centralized SCA algorithm requires a large amount of time to converge to the local optimal result, and when the scale of the problem gets larger, the dimension of the optimization will increase exponentially, and the computation time will become incredibly long. One solution is to propose a parallel algorithm to compute the result in a 'multi-thread' way. In fact, we have implemented a ADMM solution for parallel computing, but due to the scale of this course project, we have to terminate our research here. In the future, nevertheless, we will study on these issues in order to find new approaches to reduce the computation overhead and enhance the robustness of the algorithm. 
