import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
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


experiments = {0: {"name": "fetch_Reach", "gym_name": 'FetchReachDense-v2', "sim_name": 'FetchReach-v2', "state_dim": 10, "goal_dim": 3, "action_dim": 4},
               1: {"name": "fetch_Slide", "gym_name": 'FetchSlideDense-v2', "sim_name": 'FetchSlide-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4},
               2: {"name": "fetch_Push", "gym_name": 'FetchPushDense-v2', "sim_name": 'FetchPush-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4}}

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
parser.add_argument('--exploration', default=1, type=int) # from 1 to 50
parser.add_argument('--environment', default=0, type=int)
parser.add_argument('--algorithm', default=9, type=int)
parser.add_argument('--use_images', default=1, type=int)

''' GeometricRL HyperParameters '''
parser.add_argument('--policy_type', default=1, type=int)  # -1: PHI, 0:DDPG, 1: REINFORCE, 2: PPO
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--K', default=5, type=int, help="number of PPO update steps")
parser.add_argument('--var', default=1.0, type=float)
parser.add_argument('--R_gamma', default=1.0, type=float)
parser.add_argument('--reg', default=1.0, type=float, help="contrastive negative") #TODO: 0.1

parser.add_argument('--pi_clip', default=-1.0, type=float, help="clip grad policy")

''' Policy Architecture'''
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--hidden_units', default=32, type=int) # 32
parser.add_argument('--activation', default=0, type=int, help="0: relu, 1: tanh")
parser.add_argument('--projection', default=0, type=int, help="0: concat, 1: proj 2 layers")

''' Baselines HyperParameters '''
parser.add_argument('--n_critics', default=2, type=int)
parser.add_argument('--n_actions', default=2, type=int)
parser.add_argument('--conservative_weight', default=5.0, type=float) # TODO: 1.0
parser.add_argument('--expectile', default=0.8, type=float)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False
if args.use_images == 1:
    device_flag = torch.cuda.is_available()

''' HyperParameters '''

exploration_strategy = args.exploration
environment_details = experiments[args.environment]

EPOCHS = 100
batches_per_epoch = 50 #500
gamma = 0.95

input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]  # 4, 2, 2
z_dim = args.z_dim
#n_layers = 4
K = args.K
var = args.var
R_gamma = args.R_gamma
reg = args.reg
batch_size = args.batch_size

use_images = args.use_images

policy_type = args.policy_type
pi_clip = args.pi_clip if args.pi_clip > 0 else None

network_def = {'n_layers': args.n_layers, 'hidden_units': args.hidden_units, 'activation': args.activation, 'projection': args.projection}
baseline_hyper = {'n_critics': args.n_critics, 'n_actions': args.n_actions, 'conservative_weight': args.conservative_weight, 'expectile': args.expectile}

algo_id = args.algorithm

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


for exploration_strategy in [1, 10, 20, 30, 40, 50]:
    for algo_id in [0, 3, 4, 5, 7, 8, 9]:

        algo_name = algorithms[algo_id]

        exp_name = environment_details["name"] + "_R=-" + str(exploration_strategy)
        exp_name += "_" + algo_name + "_BS=" + str(batch_size) + "_use_images=" + str(use_images)
        if algo_name == 'GeometricRL':
            exp_name += "_z_dim=" + str(z_dim) #"_policy_type=" + str(policy_type) + "_K=" + str(K)
            exp_name += "_reg=" + str(reg) #"_R_gamma=" + str(R_gamma) + "_pi_clip=" + str(pi_clip)
        else:
            exp_name += "_baselines_hyper=" + str(baseline_hyper)

        exp_name += "_seed=" + str(seed)

        writer = SummaryWriter("./logs/" + exp_name)

        ''' Dataset and simulator '''
        if use_images == 1:
            env = gym.make(environment_details["sim_name"], render_mode='rgb_array')
        else:
            env = gym.make(environment_details["sim_name"])

        file_name = "./datasets_var/" + environment_details["name"] + "/" + "R=-" + str(exploration_strategy) + ".npz"
        filenpz = np.load(file_name)
        states, images, images_goal, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5'], filenpz['arr_6']

        if algo_id == 0 or algo_id == 10:
            dataset = Dataset_Uniform_precollected(states, actions, rewards, terminations, next_states, device, batches_per_epoch*batch_size, input_dim, images=images, images_goal=images_goal)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        elif algo_id == 9:
            dataset = Dataset_Visitation((states, actions, rewards, terminations, next_states), None, device, batches_per_epoch * batch_size, gamma, obs_shape=input_dim, images=images, images_goal=images_goal)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        else:
            if use_images == 0:
                dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminations)
            else:
                dataset = d3rlpy.dataset.MDPDataset(images.astype(np.uint8), actions, rewards, terminations)
            dataloader = None


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
                                device=device,
                                use_images=use_images).to(device)

            train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type, use_images=use_images)

        # elif algo_name == 'QuasiMetric':
        #     algo_class = None
        #     agent = QuasiMetric(input_dim, goal_dim, z_dim, a_dim, R_gamma, network_def, var, device).to(device)
        #
        #     train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type)

        elif algo_name == 'ContrastiveRL':
            algo_class = None
            offline_reg = 0.05
            agent = ContrastiveRL(input_dim, goal_dim, z_dim, a_dim, network_def, var, offline_reg, device, use_images=use_images).to(device)

            train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type, use_images=use_images)

        else:
            algo_class, agent = get_algo(algo_name, gamma, batch_size, device_flag, network_def, baseline_hyper, use_images=use_images)
            sparse_reward_records = []
            dense_norm_reward_records = []

            def callback(algo: algo_class, epoch: int, total_step: int) -> None:  # d3rlpy.algos.QLearningAlgoBase
                avg_sparse_reward, avg_dense_score = simulation(agent, env, device, args.environment, environment_details, render=False, n_episodes=100, use_images=use_images, image_rescale=True if use_images == 1 else False)
                writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch-1)
                writer.add_scalar("Rewards/dense_score", avg_dense_score, epoch-1)
                sparse_reward_records.append(avg_sparse_reward)
                dense_norm_reward_records.append(avg_dense_score)
                np.savez("./saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

            agent.fit(dataset,
                      n_steps=batches_per_epoch*EPOCHS,
                      n_steps_per_epoch=batches_per_epoch,
                      experiment_name=None,
                      save_interval=batches_per_epoch*EPOCHS+1,
                      epoch_callback=callback,
                      )


        writer.close()












