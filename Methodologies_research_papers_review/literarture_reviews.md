1. Recent Advances in Robot
Learning from Demonstration (https://www.annualreviews.org/content/journals/10.1146/annurev-control-100819-063206)
-> How demonstration is obtained: Kinesthetic teaching, teleoperation, or passive observation of a human performing the task. 
What the robot learns: A policy (how to act), a cost/reward function (why to act), or a task plan (what sequence of actions to perform). 
2. Keyframe-based Learning from Demonstration (https://link.springer.com/article/10.1007/s12369-012-0160-0)
-> Overall used kinesthetic teaching. KLfD learns robot skills from trajectory, keyframe, or hybrid demonstrations by converting them into keyframes, aligning and clustering them into Sequential Pose Distributions (SPD), and generating a smooth trajectory between the learned poses. Compared with traditional trajectory-based methods, it allows the teacher to use sparse keyframes reducing unwanted/noisy movements and making teaching easier; trajectory demonstrations are better for skills with important speed or curved-motion information, keyframes suit simple/straight or point-to-point motions, and hybrid demonstrations suit skills containing both types of motion. 
3. Rob ot Learning From Demonstration (https://mcgovern-fagg.org/amy_html/courses/cs5973_fall2005/lfd.pdf)
-> Overall used passive observation. The human hands dynamics may be different from robot's. * Non-parametric: Learns the relationship directly from the data. Kalman Filter There is about 0.12 s sensing delay The Kalman filter predicts the future state to compensate for this delay.
4. A survey of demonstration learning (https://www.sciencedirect.com/science/article/pii/S0921889024001969?ref=cra_js_challenge&fr=RR-1)
->Overall pipeline: Choose the demonstrator → demonstration technique → data representation → learning method → optimization → evaluation. Learning methods: The paper separates learning into Policy learning, Inverse RL, Model learning, Apprenticeship, GAIL, and Sequence Models.
5. A Comprehensive Study on Robot Learning
from Demonstration (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9074946&tag=1)
-> Learning stages include these two: Low-level skills: Learn individual motions or trajectories. Methods can include DMP, GMM-GMR, and HMM.
High-level skills: Learn the task plan, action sequence, or goal/objective.
6. A robot learning from demonstration framework
to perform force-based manipulation tasks
-> Demonstrations → HMM learns the task structure → GMRa uses the learned sequential model to generate actions during execution. HMM encodes the demonstrations as sequential hidden states, capturing perceptions, actions, and temporal structure. GMRa uses the trained HMM and current/past perceptions to generate the appropriate robot actions.
7. Robot Learning from Demonstration: A Task-level Planning Approach (https://journals.sagepub.com/doi/full/10.5772/5611)
-> learning high-level manipulation tasks from demonstrations, robot does not store and replay the human hand trajectory. Instead, it learns what was done, represents the task using states and constraints, and then uses a task planner to generate a new action sequence according to the current environment. 
POSE DETECTION: Camera image → SIFT feature detection → Match detected features with stored object features → Estimate object pose.
The complete demonstration is divided into isolated operations or primitive tasks. After segmentation, each subtask is represented as a state describing its effect on the world.
Once the task model is available, the robot uses a task planner to find a sequence of actions that reaches the desired goal.
The planner is inspired by STRIPS and represents actions using preconditions and effects.
Before planning, the system checks possible collisions.
8. Integrating Object and Grasp Recognition for
Dynamic Scene Interpretation (https://ieeexplore.ieee.org/abstract/document/1507432)
-> Passive observation, Object detection: Color Cooccurrence Histogram (CCH) matching. Object localization: CCH vote matrix → estimate object center → convert image coordinates to world coordinates
9. Robot Learning from Demonstration in Robotic
Assembly: A Survey
->Utilizes automatic reverse execution to revert the robot to a safe state when errors occur before resuming forward assembly. Evaluates skill reproduction using a Weighted Similarity Measure (Euclidean distance with covariance), a Generic Similarity Measure (joint, path, and distance constraints), and Combined Metrics (RMS errors, Dynamic Time Warping, and acceleration derivation for smoothness)
10. A survey of robot learning from demonstration (https://www.sciencedirect.com/science/article/pii/S0921889008001772)
-> kinesthetic teaching operates with direct identity mappings; because the human teacher physically moves the robot's passive joints while the robot records from its own sensors, it inherently eliminates both Record Mapping and Embodiment Mapping differences.
11. Imitation Learning (https://web.stanford.edu/class/cs237b/pdfs/lecture/lecture_10111213.pdf)
-> a brief about how can you mimic, learn the underlying funds + effiecnt behavoir, learning the main goal (Intent inference)
12. Inverse Reinforcement Learning Framework for Transferring Task
Sequencing Policies from Humans to Robots in Manufacturing Applications (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160687)
->  four main IRL approaches: Max-margin IRL, Max-entropy IRL, Bayesian     xIRL, and Regression/Classification IRL. They did not use max entropy or bayesian for they wish to trace human demonstration somewhere strictly.
13. Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware (https://arxiv.org/pdf/2304.13705)
-> finally ACT policy has be choosen by Animesh because of its beginner freindly easeness.