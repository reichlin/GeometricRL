import os
import gymnasium as gym
import minari
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse

from models.geometricRL import GeometricRL
from models.quasimetric import QuasiMetric
from models.contrastive_occupancy import ContrastiveRL
from my_utils.model_utils import *
from my_utils.torch_datasets import Dataset_Minari, Dataset_Uniform_precollected, Dataset_Visitation
from torch.utils.data import DataLoader
import d3rlpy
import dataclasses
from my_utils.algorithms_utils import *


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

algorithms = {0: 'GeometricRL',
              1: 'DDPG',
              2: 'BC',
              3: 'CQL',  # yes
              4: 'BCQ',  # maybe
              5: 'BEAR',  # yes
              6: 'AWAC',  # no
              7: 'PLAS',  # yes
              8: 'IQL',  # no
              9: 'ContrastiveRL',
              10: 'QuasiMetric',
              11: 'Diffuser',
              }

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--seed', default=0, type=int)

''' Dataset and Algorithm '''
parser.add_argument('--exploration', default=0, type=int)  # 0: uniform, 1: OU - process, 2: minari
parser.add_argument('--environment', default=2, type=int)
parser.add_argument('--algorithm', default=9, type=int)

''' GeometricRL HyperParameters '''
parser.add_argument('--policy_type', default=1, type=int)  # -1: PHI, 0:DDPG, 1: REINFORCE, 2: PPO
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--K', default=5, type=int, help="number of PPO update steps")
parser.add_argument('--var', default=1.0, type=float)
parser.add_argument('--R_gamma', default=3.0, type=float)
parser.add_argument('--reg', default=1.0, type=float, help="contrastive negative") #TODO: 0.1

parser.add_argument('--pi_clip', default=-1.0, type=float, help="clip grad policy")

''' Policy Architecture'''
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--hidden_units', default=32, type=int) # 32
parser.add_argument('--activation', default=0, type=int, help="0: relu, 1: tanh")
parser.add_argument('--projection', default=0, type=int, help="0: concat, 1: proj 2 layers")

''' Baselines HyperParameters '''
parser.add_argument('--n_critics', default=2, type=int)
parser.add_argument('--n_actions', default=10, type=int)
parser.add_argument('--conservative_weight', default=1.0, type=float) # TODO: 1.0
parser.add_argument('--expectile', default=0.7, type=float)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False

''' HyperParameters '''

exploration_strategy = exploration_processes[args.exploration]
environment_details = experiments[args.environment]

EPOCHS = 100
batches_per_epoch = 1000
gamma = 0.95

input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]  # 4, 2, 2
z_dim = args.z_dim
#n_layers = 4
K = args.K
var = args.var
R_gamma = args.R_gamma
reg = args.reg
batch_size = args.batch_size

policy_type = args.policy_type
pi_clip = args.pi_clip if args.pi_clip > 0 else None

network_def = {'n_layers': args.n_layers, 'hidden_units': args.hidden_units, 'activation': args.activation, 'projection': args.projection}
baseline_hyper = {'n_critics': args.n_critics, 'n_actions': args.n_actions, 'conservative_weight': args.conservative_weight, 'expectile': args.expectile}

algo_id = args.algorithm
algo_name = algorithms[algo_id]

exp_name = environment_details["name"] + "_" + exploration_strategy
exp_name += "_" + algo_name + "_BS=" + str(batch_size)
if algo_name == 'GeometricRL':
    exp_name += "_z_dim=" + str(z_dim) #"_policy_type=" + str(policy_type) + "_K=" + str(K)
    exp_name += "_reg=" + str(reg) #"_R_gamma=" + str(R_gamma) + "_pi_clip=" + str(pi_clip)
else:
    exp_name += "_baselines_hyper=" + str(baseline_hyper)

exp_name += "_seed=" + str(seed)

writer = SummaryWriter("./logs/" + exp_name)

''' Dataset and simulator '''
#max_T = 300  # default is 300
env = gym.make(environment_details["gym_name"], continuing_task=False)  # 'PointMaze_UMaze-v3' max_episode_steps=max_T
if args.exploration < 2:  # pre-collected

    if args.environment < 3:
        file_name = "./datasets/" + exploration_strategy + "_" + environment_details["name"] + "_" + str(0) + ".npz"
        filenpz = np.load(file_name)
        states, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4']
    else:
        states, actions, rewards, terminations, next_states = [], [], [], [], []
        for id in range(10):
            file_name = "./datasets/" + exploration_strategy + "_" + environment_details["name"] + "_" + str(id) + ".npz"
            filenpz = np.load(file_name)
            tmp_states, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4']
            states.append(tmp_states)
            actions.append(tmp_actions)
            rewards.append(tmp_rewards)
            terminations.append(tmp_terminations)
            next_states.append(tmp_next_states)
        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        rewards = np.concatenate(rewards, 0)
        terminations = np.concatenate(terminations, 0)
        next_states = np.concatenate(next_states, 0)

    if algo_id == 0 or algo_id == 10:
        dataset = Dataset_Uniform_precollected(states, actions, rewards, terminations, next_states, device, batches_per_epoch*batch_size, input_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #batches_per_epoch = int(len(dataset) / batch_size)
    elif algo_id == 9:
        dataset = Dataset_Visitation((states, actions, rewards, terminations, next_states), None, device, batches_per_epoch * batch_size, gamma, obs_shape=input_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    else:
        dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminations)
        dataloader = None
        #batches_per_epoch = int(dataset.transition_count / batch_size)

else:  # minari
    try:
        offline_dataset = minari.load_dataset(environment_details["minari_name"]) #"pointmaze-umaze-v1")
    except:
        minari.download_dataset(environment_details["minari_name"]) #"pointmaze-umaze-v1")
        offline_dataset = minari.load_dataset(environment_details["minari_name"]) #"pointmaze-umaze-v1")

    if algo_id == 0 or algo_id == 10:
        dataset = Dataset_Minari(offline_dataset, device, batches_per_epoch*batch_size, include_achiev_goal=(args.environment>2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #batches_per_epoch = int(len(dataset) / batch_size)
    elif algo_id == 9:
        dataset = Dataset_Visitation(None, offline_dataset, device, batches_per_epoch * batch_size, gamma, include_achiev_goal=(args.environment>2))
        dataloader = DataLoader(dataset, batch_size=batch_size)
    else:

        states, actions, rewards, terminations = None, None, None, None
        for e in tqdm(offline_dataset.episode_indices):
            trj = offline_dataset[e]
            if trj.observations['observation'].shape[0] < 2 or trj.actions.shape[0] < 1:
                print("empty trj", e)
            elif (trj.observations['observation'].shape[0] - 1) == trj.actions.shape[0]:
                if args.environment > 2:
                    obs = np.concatenate((trj.observations['achieved_goal'], trj.observations['observation'], trj.observations['desired_goal']), -1)[:-1]
                else:
                    obs = np.concatenate((trj.observations['observation'], trj.observations['desired_goal']), -1)[:-1]
                states = obs if states is None else np.concatenate((states, obs), 0)
                actions = trj.actions if actions is None else np.concatenate((actions, trj.actions), 0)
                rewards = trj.rewards if rewards is None else np.concatenate((rewards, trj.rewards), 0)
                terminations = trj.terminations*1. if terminations is None else np.concatenate((terminations, trj.terminations*1.), 0)
                terminations[-1] = 1.
            else:
                print("non complete trj", e)

        dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminations)
        dataloader = None
        #batches_per_epoch = int(dataset.transition_count / batch_size)



''' Algorithm '''
if algo_name == 'GeometricRL':
    algo_class = None
    agent = GeometricRL(input_dim,
                        goal_dim,
                        z_dim,
                        a_dim,
                        gamma,
                        network_def=network_def,
                        K=K,
                        var=var,
                        R_gamma=R_gamma,
                        reg=reg,
                        policy_type=policy_type,
                        policy_clip=pi_clip,
                        device=device).to(device)

    train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type)

elif algo_name == 'QuasiMetric':
    algo_class = None
    agent = QuasiMetric(input_dim, goal_dim, z_dim, a_dim, R_gamma, network_def, var, device).to(device)

    train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type)

elif algo_name == 'ContrastiveRL':
    algo_class = None
    offline_reg = 0.05
    agent = ContrastiveRL(input_dim, goal_dim, z_dim, a_dim, network_def, var, offline_reg, device).to(device)

    train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type)

else:
    algo_class, agent = get_algo(algo_name, gamma, batch_size, device_flag, network_def, baseline_hyper)
    sparse_reward_records = []
    dense_norm_reward_records = []

    def callback(algo: algo_class, epoch: int, total_step: int) -> None:  # d3rlpy.algos.QLearningAlgoBase
        avg_sparse_reward, avg_dense_score = simulation(agent, env, device, args.environment, environment_details, render=False, n_episodes=100)
        writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch-1)
        writer.add_scalar("Rewards/dense_score", avg_dense_score, epoch-1)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez("./saved_results/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

    agent.fit(dataset,
              n_steps=batches_per_epoch*EPOCHS,
              n_steps_per_epoch=batches_per_epoch,
              experiment_name=None,
              save_interval=batches_per_epoch*EPOCHS+1,
              epoch_callback=callback,
              )


writer.close()












