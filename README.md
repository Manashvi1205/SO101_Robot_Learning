# SO-101 Robot Learning

This repository contains my work on **robot learning and imitation learning using the SO-101 robotic arm**. The project focuses on enabling a robot to learn manipulation tasks from demonstrations rather than relying on manually specified joint trajectories or inverse-kinematics-based motion programming.

## Project Overview

The initial objective is to teach an SO-101 robotic arm to perform manipulation tasks from demonstrations collected using a leader-follower setup.

The first task considered in this project is a simple pick-and-place task:

> **Pick up a box from the start area and place it in the finish area.**

The project will begin with basic demonstration-based learning and imitation learning, and gradually explore more advanced robot-learning techniques.

## Problem Statement

Traditional robotic manipulation often requires manually specifying robot poses, trajectories, waypoints, or inverse kinematics.

The goal of this project is to investigate whether an SO-101 arm can instead **learn a manipulation behavior directly from demonstrations**, using recorded robot actions and visual observations.

The broader objective is to understand how demonstrations can be converted into a learned policy that can reproduce the desired task autonomously.

## Hardware

- **SO-101 Leader Arm** — used to provide demonstrations
- **SO-101 Follower Arm** — used to execute the demonstrated motions
- **Wrist-mounted camera** — used to record visual observations
- Computer for robot control, data collection, and model training

The leader and follower arms are operated using the **LeRobot** framework.

## Software

- Python
- LeRobot
- Git / GitHub
- Linux / Ubuntu
- ROS 2 and other robotics tools where required

## Data Collection

Demonstrations are collected using the leader-follower setup.

During a demonstration:

1. The operator moves the leader SO-101.
2. The follower SO-101 reproduces the leader's movements.
3. Robot observations and actions are recorded.
4. Camera observations are recorded using the wrist-mounted camera.
5. The resulting demonstration is stored as a dataset for robot-learning experiments.

The initial dataset contains demonstrations for a box pick-and-place task.

## Current Task

### Pick and Place

The current experimental setup consists of:

- A designated **start area**
- A designated **finish area**
- A box that must be picked up
- An SO-101 robotic arm with a wrist-mounted camera

The desired behavior is:

```text
Box at Start
     ↓
Approach box
     ↓
Grasp box
     ↓
Lift box
     ↓
Move to Finish
     ↓
Release box
     ↓
Box at Finish
```

## Learning Pipeline

The planned learning pipeline is:

```text
Demonstrations
      ↓
Dataset Collection
      ↓
Dataset Inspection
      ↓
Data Preprocessing
      ↓
Imitation Learning
      ↓
Learned Policy
      ↓
Evaluation on SO-101
      ↓
Further Robot-Learning Methods
```

The project will start with simple learning methods to establish a baseline before moving toward more advanced approaches.

## Planned Experiments

The project will investigate questions such as:

- How much demonstration data is required to learn a task?
- How does demonstration quality affect the learned behavior?
- Can visual observations be used to learn the manipulation task?
- How well does a learned policy generalize to changes in object position?
- How does the learned policy compare with the original human demonstrations?
- Which robot-learning methods are suitable for the SO-101 platform?

## Repository Structure

The repository is currently based on an existing robotics codebase and will be reorganized as the project develops.

The intended structure is approximately:

```text
SO101_Robot_Learning/
│
├── data/              # Demonstration datasets
├── scripts/            # Data collection and utility scripts
├── models/             # Trained models and checkpoints
├── configs/            # Experiment configurations
├── notebooks/          # Dataset analysis and experiments
├── results/            # Evaluation results
│
├── README.md
└── requirements.txt
```

The structure may change as the project evolves.

## Current Status

**Current stage:**

- SO-101 leader and follower arms configured
- Both arms calibrated
- Leader-follower teleoperation working
- Wrist camera configured
- Initial pick-and-place demonstration data collected
- Beginning the robot-learning and dataset-analysis stage

## Future Work

- Collect a larger demonstration dataset
- Analyze and visualize the recorded observations and actions
- Establish a baseline imitation-learning model
- Train a policy using the collected demonstrations
- Evaluate the learned policy on the SO-101
- Test generalization to different object positions and configurations
- Investigate different robot-learning approaches
- Compare the performance of different learning methods
- Gradually move toward more complex manipulation tasks

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- Hugging Face robotics and robot-learning resources
- Research literature on imitation learning, learning from demonstrations, and robot manipulation

## Author

**Manashvi Garg**  
Electrical Engineering, IIT Delhi

This repository documents the development and experimentation carried out as part of my work on robot learning with the SO-101 robotic arm.
