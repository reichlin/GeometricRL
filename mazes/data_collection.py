import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import minari
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--name_id', default=0, type=int)
parser.add_argument('--expl', default=1, type=int)
parser.add_argument('--env', default=0, type=int)
args = parser.parse_args()

exploration_processes = {0: "Random",
                         1: "Ornstein-Uhlenbeck",
                         2: "Minari"}

# name Minari, name Gym, state space dim, goal space dim, action space dim
experiments = {0: {"name": "point_uMaze", "minari_name": 'pointmaze-umaze-v1', "gym_name": 'PointMaze_UMaze-v3', "d4rl_scores": [23.85, 161.86], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               1: {"name": "point_Medium", "minari_name": 'pointmaze-medium-v1', "gym_name": 'PointMaze_Medium_Diverse_GR-v3', "d4rl_scores": [13.13, 277.39], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               2: {"name": "point_Large", "minari_name": 'pointmaze-large-v1', "gym_name": 'PointMaze_Large_Diverse_GR-v3', "d4rl_scores": [6.7, 273.99], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               3: {"name": "ant_uMaze", "minari_name": 'antmaze-umaze-diverse-v0', "gym_name": 'AntMaze_UMaze-v4', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8},
               4: {"name": "ant_Medium", "minari_name": 'antmaze-medium-diverse-v0', "gym_name": 'AntMaze_Medium_Diverse_GR-v4', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8},
               5: {"name": "ant_Large", "minari_name": 'antmaze-large-diverse-v0', "gym_name": 'AntMaze_Large_Diverse_GR-v4', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8}}

theta = 0.1
sigma = 0.2
exploration_strategy = exploration_processes[args.expl]

# for expl_id in range(2):
#     exploration_strategy = exploration_processes[expl_id]
#     for id in range(6):

minari.download_dataset(dataset_id=experiments[args.env]["minari_name"])

max_T = 1000000
if args.env < 3:
    env = gym.make(experiments[args.env]["gym_name"], continuing_task=False)
    total_episodes = 10000
else:
    env = gym.make(experiments[args.env]["gym_name"], continuing_task=False, max_episode_steps=max_T)
    total_episodes = 10



states, actions, rewards, terminations, next_states = None, None, None, None, None
for _ in tqdm(range(total_episodes)):
    tmp_states, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], []
    obs, _ = env.reset()
    if args.env < 3:
        st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
    else:
        st = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
    at = env.action_space.sample()
    for t in range(max_T):
        if exploration_strategy == "Ornstein-Uhlenbeck":
            zt = env.action_space.sample()
            at = (1-theta)*at + sigma*zt
        else:
            at = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(at)
        tmp_states.append(st)
        tmp_actions.append(np.expand_dims(at, 0))
        tmp_rewards.append(np.array([[r]]))
        tmp_terminations.append(np.array([[(term or trunc) * 1.]]))
        if args.env < 3:
            st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
        else:
            st = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
        tmp_next_states.append(st)
        if term or trunc:
            break

    if args.env >= 3:
        idx = np.random.choice(range(len(tmp_states)), int(max_T*0.01))
    else:
        idx = np.arange(len(tmp_states))
    tmp_states = np.concatenate(tmp_states, 0)[idx]
    tmp_actions = np.concatenate(tmp_actions, 0)[idx]
    tmp_rewards = np.concatenate(tmp_rewards, 0)[idx]
    tmp_terminations = np.concatenate(tmp_terminations, 0)[idx]
    tmp_next_states = np.concatenate(tmp_next_states, 0)[idx]
    if states is None:
        states = tmp_states
        actions = tmp_actions
        rewards = tmp_rewards
        terminations = tmp_terminations
        next_states = tmp_next_states
    else:
        states = np.concatenate((states, tmp_states), 0)
        actions = np.concatenate((actions, tmp_actions), 0)
        rewards = np.concatenate((rewards, tmp_rewards), 0)
        terminations = np.concatenate((terminations, tmp_terminations), 0)
        next_states = np.concatenate((next_states, tmp_next_states), 0)

    # terminations[-1][0, 0] = 1.
    terminations[-1, 0] = 1.

file_name = "./datasets/" + exploration_strategy + "_" + experiments[args.env]["name"] + "_" + str(args.name_id) + ".npz"
# np.savez(file_name, np.concatenate(states, 0), np.concatenate(actions, 0), np.concatenate(rewards, 0), np.concatenate(terminations, 0), np.concatenate(next_states, 0))
np.savez(file_name, states, actions, rewards, terminations, next_states)

# states, actions, rewards, terminations = None, None, None, None
# for _ in tqdm(range(total_episodes)):
#     obs, _ = env.reset()
#     if args.env < 3:
#         st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
#     else:
#         st = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
#     at = env.action_space.sample()
#     for t in range(max_T):
#         if exploration_strategy == "Ornstein-Uhlenbeck":
#             zt = env.action_space.sample()
#             at = (1-theta)*at + sigma*zt
#         else:
#             at = env.action_space.sample()
#         obs, r, term, trunc, _ = env.step(at)
#         states = st if states is None else np.concatenate((states, st), 0)
#         actions = np.expand_dims(at, 0) if actions is None else np.concatenate((actions, np.expand_dims(at, 0)), 0)
#         rewards = np.array([[r]]) if rewards is None else np.concatenate((rewards, np.array([[r]])), 0)
#         terminations = np.array([[(term or trunc) * 1.]]) if terminations is None else np.concatenate((terminations, np.array([[(term or trunc) * 1.]])), 0)
#         if args.env < 3:
#             st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
#         else:
#             st = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
#     terminations[-1, 0] = 1.
#
# file_name = "./datasets/" + exploration_strategy + "_" + experiments[args.env][0] + ".npz"
# np.savez(file_name, states, actions, rewards, terminations)







