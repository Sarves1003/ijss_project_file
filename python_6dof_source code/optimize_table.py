#!/usr/bin/env python3
"""
Creates a publication-grade, compact literature comparison table (tables/comparison_table.tex)
that formats cleanly within 2-3 pages so the total manuscript length targets 30-35 pages.
"""

table_tex = r"""% Publication-Grade Literature Comparison Table for Q1 Journal Target
\begingroup
\footnotesize
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\begin{longtable}{p{2.2cm}p{0.7cm}p{2.0cm}p{2.1cm}p{1.1cm}p{2.1cm}p{2.7cm}p{2.7cm}}
\caption{Comparative analysis of representative vision-aware manipulation and control literature.}\label{tab:literature-comparison}\\
\toprule
\textbf{Reference} & \textbf{Year} & \textbf{Platform} & \textbf{Methodology} & \textbf{Vision} & \textbf{Controller} & \textbf{Primary Advantage} & \textbf{Limitation / Open Gap} \\
\midrule
\endfirsthead
\multicolumn{8}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
\textbf{Reference} & \textbf{Year} & \textbf{Platform} & \textbf{Methodology} & \textbf{Vision} & \textbf{Controller} & \textbf{Primary Advantage} & \textbf{Limitation / Open Gap} \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot

Denavit \cite{Denavit1955} & 1955 & Lower pairs & Kinematic Matrix & No & Open Loop & Standardized 4-parameter notation & Lacks dynamics and compliance \\
Yoshikawa \cite{Yoshikawa1985} & 1985 & Serial Arm & Manipulability Index & No & Task Space & Quantitative velocity isotropy & Ignores joint torque bounds \\
Luh et al. \cite{Luh1980} & 1980 & Industrial Arm & Newton-Euler Recursive & No & Joint Servo & Real-time $O(N)$ computational speed & Requires full inertial parameters \\
Slotine \& Li \cite{Slotine1987} & 1987 & Manipulator & Adaptive Control & No & Passivity-Based & Asymptotically stable under inertia errors & High computational regressor load \\
Utkin \cite{Utkin1993} & 1993 & Electric Drive & Sliding Mode Control & No & Discontinuous & Invariant to matched disturbances & High-frequency chattering \\
Zhang \cite{Zhang2000} & 2000 & Camera Plane & Planar Calibration & 2D & Open Loop & Flexible checkerboard technique & Requires manual target placement \\
Tsai \cite{Tsai1987} & 1987 & TV Camera & Hand-Eye Calibration & 2D & Open Loop & Closed-form radial distortion model & Sensitive to feature detection noise \\
Horaud \cite{Horaud1995} & 1995 & Hand-Eye & Non-linear Optimization & 2D & Visual Servo & Solves $AX=XB$ explicitly & Sensitive to initialization \\
Siciliano \cite{Siciliano2009} & 2009 & General Arm & Operational Space & No & Inverse Dynamics & Comprehensive control methodology & High mathematical complexity \\
Craig \cite{Craig2018} & 2018 & Manipulator & Mechanics \& Control & No & PID / CTC & Fundamental robotic baseline & Standard linear assumptions \\
Spong et al. \cite{Spong2020} & 2020 & Serial Arm & Euler-Lagrange & No & Robust / Adaptive & Rigorous passivity guarantees & Model mismatch sensitivity \\
Astrom \cite{Astrom2006} & 2006 & General System & PID Tuning & No & Classical PID & Benchmark industry baseline & Poor nonlinear tracking \\
Armstrong \cite{Armstrong1994} & 1994 & Machine Tool & Stribeck Friction & No & Friction Comp. & Models low-velocity stick-slip & Non-differentiable zero crossing \\
Makkar \cite{Makkar2007} & 2007 & Dynamic System & Smooth Friction & No & Smooth Comp. & Continuous $C^\infty$ friction model & Empirical parameter fitting \\
Featherstone \cite{Featherstone2008} & 2008 & Rigid Body & Spatial Vectors & No & Dynamics Algo & Highly efficient algorithm & Abstract mathematical notation \\
Camacho \cite{Camacho2013} & 2013 & Process & Model Predictive & No & Constrained MPC & Explicit constraint handling & High online optimization cost \\
Lewis et al. \cite{Lewis2012} & 2012 & State System & LQR / CARE & No & Optimal Linear & Guaranteed gain/phase margins & Requires linear state-space \\
Zhou \& Doyle \cite{Zhou1998} & 1998 & Linear Plant & $H_\infty$ Robust & No & Minimax Robust & Worst-case disturbance rejection & Conservative gain selection \\
Levenberg \cite{Levenberg1944} & 1944 & Least Squares & Damped Gauss & No & Optimization & Solves ill-conditioned IK & Requires damping selection \\
Marquardt \cite{Marquardt1963} & 1963 & Least Squares & Adaptive Damping & No & Optimization & Smooth interpolation LM algorithm & Local minima susceptibility \\
Nakamura \cite{Nakamura1986} & 1986 & Manipulator & Singularity Damping & No & DLS IK & Prevents infinite joint velocity & Introduces position tracking error \\
Suzuki \cite{Suzuki1985} & 1985 & Binary Image & Border Following & 2D & Vision & Fast contour topological analysis & Sensitive to binary thresholding \\
Quigley \cite{Quigley2009} & 2009 & Middleware & ROS Framework & No & Distributed & Modular publish-subscribe system & Network latency overhead \\
Levine et al. \cite{Levine2018} & 2018 & PR2 Arm & Deep RL / Visuomotor & 2D & End-to-End & End-to-end motor policy learning & Massive sample complexity \\
Mahler et al. \cite{Mahler2017} & 2017 & Industrial Arm & Dex-Net / GraspNet & 3D & Quality Metric & Synthetic training dataset & Sim-to-real transfer gap \\
Peng et al. \cite{Peng2022} & 2022 & Collaborative & Vision Pick-Place & 2D/3D & Task Control & Survey on visual manipulation & Lacks experimental benchmark \\
Zeng et al. \cite{Zeng2021} & 2022 & Robot Arm & Multi-Affordance & 3D & Deep Learning & Clutter pick-and-place grasping & High computational demand \\
Xiang et al. \cite{Xiang2018} & 2018 & RGB-D & PoseCNN & 3D & Pose Estimation & 6D pose estimation in clutter & High GPU requirements \\
Wang et al. \cite{Wang2019} & 2019 & RGB-D & DenseFusion & 3D & Pose Fusion & Pixel-wise geometry fusion & Sensitive to reflective objects \\
Gualtieri \cite{Gualtieri2021} & 2021 & Cobot Workcell & SME Collaboration & No & Safety Control & Human-robot safety standards & Limited to light payloads \\
Hjorth \cite{Hjorth2023} & 2022 & Smart Factory & Industrial HRC & No & Adaptive HRC & Survey on collaborative robotics & Qualitative positioning \\
Li \& Zhang \cite{Li2022} & 2022 & UR10 Cobot & Robust $H_\infty$ & No & Disturbance Rej. & Load variation robustness & Complex weight selection \\
Chen et al. \cite{Chen2020} & 2016 & Industrial Plant & Disturbance Observer & No & DOB Control & Active disturbance compensation & Requires accurate plant model \\
Kennedy \cite{Kennedy1995} & 1995 & Swarm & PSO Algorithm & No & Metaheuristic & Fast global search velocity & Premature convergence \\
Mirjalili \cite{Mirjalili2014} & 2014 & Swarm & Grey Wolf (GWO) & No & Metaheuristic & High exploration & Parameter tuning required \\
Holland \cite{Holland1992} & 1992 & Evolution & Genetic Algorithm & No & GA Tuning & Robust discrete search & Slow convergence rate \\
Goldberg \cite{Goldberg1989} & 1989 & Evolution & Genetic Search & No & GA Optimization & Classic evolutionary benchmark & Scalability limitations \\
Das \cite{Das2016} & 2011 & Differential & Differential Evolution & No & DE Optimization & Excellent continuous optimization & Parameter sensitivity \\
Haarnoja \cite{Haarnoja2018} & 2018 & RL Agent & Soft Actor-Critic & No & Max-Entropy RL & Off-policy stability and exploration & Hyperparameter sensitivity \\
Sutton \cite{Sutton2018} & 2018 & RL Framework & Reinforcement Learning & No & Policy Gradient & Fundamental RL foundation & High sample variance \\
Mnih et al. \cite{Mnih2015} & 2015 & Deep Q & DQN Architecture & 2D & Deep Q-Learning & High-dimensional visual control & Discrete action space \\
Redmon \cite{Redmon2018} & 2018 & Vision CNN & YOLOv3 Detector & 2D & Real-time Vision & Fast single-stage object detection & Bounding box discretization \\
He et al. \cite{He2016} & 2016 & Deep Network & ResNet Architecture & 2D & Deep Feature & Deep residual skip connections & High inference latency \\
Duhlev \cite{Duhlev2021} & 2021 & myCobot & HSV Color Vision & 2D & PID Control & Low-cost sorting implementation & Uncompensated dynamics error \\
Elephant \cite{ElephantRobotics2023} & 2023 & myCobot 280 & 6-DOF Cobot Arm & No & Hardware Platform & Lightweight 280g cobot arm & Backlash and joint compliance \\
Zhang et al. \cite{TaylorFrancisIJSS2022} & 2022 & Collaborative & Adaptive Robust & No & Robust Tracking & Model uncertainty compensation & High gain requirements \\
Schulman \cite{Schulman2017} & 2017 & Policy RL & PPO Algorithm & No & Policy Optimization & Stable clipped surrogate objective & Sample inefficiency \\
Fujimoto \cite{Fujimoto2018} & 2018 & Actor-Critic & TD3 Algorithm & No & Continuous RL & Overestimation bias reduction & Hyperparameter sensitivity \\
\midrule
\textbf{This Work} & \textbf{2026} & \textbf{myCobot 280} & \textbf{Vision+Dynamic+Opt} & \textbf{2D Homography} & \textbf{CTC/SMC/LQR/H$_\infty$/SAC} & \textbf{Integrated math., control, and optimization rigor} & \textbf{Validated on accessible 6-DOF platform} \\
\end{longtable}
\endgroup
"""

with open("tables/comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(table_tex.strip())

with open("latex/tables/comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(table_tex.strip())

print("Updated tables/comparison_table.tex and latex/tables/comparison_table.tex.")
