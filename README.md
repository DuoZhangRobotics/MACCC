# MACCC

Multi-Agents Constrained Congestion Control

Duo Zhang, Yuhao Fan

Achieved dynamic congestion control over one with multiple senders and fairness control using Reinforcement Learning.


## Overview
This repo contains the gym environment required for training reinforcement
learning models used in the MACC project.

We re-implemented the PCC-RL with stable-baselines3 and PyTorch.

We collect the bandwidth share of each sender from the bottleneck queue of the link for the fairness control

We developed better APIs for visualization and interactions with training(tensorboard and wandb)



## Training
1. Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```
2. Activate the new environment: 
```
conda activate networks
```
3. Train: 
```
python src/gym/stable_solve.py 
```
### Arguments:

--steps, type=int, default=1600 * 410 * 50, 
Timesteps of model learning

--num_senders, type = int, default=1, 
Sender numbers

--num_links, type = int, default=1, 
Links numbers

--throughput_coefficient, type = float, default=20, 
Reward function coefficient of throughtput part

--loss_coefficient, type = float, default=1e3, 
Reward function coefficient of loss rate part

--latency_coefficient, type = float, default=2e3, 
Reward function coefficient of latency part

--fairness_coefficient, type = float, default=1e3, 
Reward function coefficient of fairness part

--PCC, type = int, default=0, 
Whether use PCC reward function

## Visualization
1. Generate the result image
```
python src/gym/graph_runall.py
```
2. Generate the compare of result image
```
src/gym/compare.sh
```
