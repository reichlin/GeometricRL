import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse

from models.PPO import PPO



experiments = {0: {"name": "fetch_Reach", "gym_name": 'FetchReachDense-v2', "sim_name": 'FetchReach-v2', "state_dim": 10, "goal_dim": 3, "action_dim": 4},
               1: {"name": "fetch_Slide", "gym_name": 'FetchSlideDense-v2', "sim_name": 'FetchSlide-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4},
               2: {"name": "fetch_Push", "gym_name": 'FetchPushDense-v2', "sim_name": 'FetchPush-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4}}

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--environment', default=1, type=int)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False

environment_details = experiments[args.environment]
input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]

env_dense = gym.make(environment_details["gym_name"])#, render_mode='human')
env_sparse = gym.make(environment_details["sim_name"])

writer = SummaryWriter("./logs_PPO/" + environment_details["name"] + "r3_entropy=0.001")


agent = PPO(input_dim+goal_dim, a_dim, 1, device)

EPOCHS = 10000000000000

for epoch in tqdm(range(EPOCHS)):
    avg_distance_to_cube = 0
    avg_distance_to_goal = 0
    for i in range(10):
        obs, _ = env_dense.reset()
        for t in range(50):
            #env.render()
            current_state = torch.from_numpy(np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)).float().to(device)

            at, logprob, sigma = agent.get_action(current_state)
            next_obs, reward_to_goal, terminated, truncated, info = env_dense.step(at[0].detach().cpu().numpy())

            training_reward = reward_to_goal
            reward_get_obj = 0
            if args.environment == 1:
                reward_get_obj = - 0.1 * np.linalg.norm(next_obs['achieved_goal'] - next_obs['observation'][:3])
            training_reward += reward_get_obj
            avg_distance_to_cube += np.linalg.norm(next_obs['achieved_goal'] - next_obs['observation'][:3])
            avg_distance_to_goal += - reward_to_goal

            done = terminated or truncated
            agent.push_batchdata(current_state.detach().cpu(), at.detach().cpu(), logprob.detach().cpu(), training_reward, done)

            obs = next_obs

            if done:
                break

    v_loss, h_loss = agent.update()
    agent.clear_batchdata()

    writer.add_scalar("Loss/v_loss", v_loss, epoch)
    writer.add_scalar("Loss/h_loss", h_loss, epoch)
    writer.add_scalar("Rewards/distance_to_cube", avg_distance_to_cube/(10*50), epoch)
    writer.add_scalar("Rewards/distance_to_goal", avg_distance_to_goal/(10*50), epoch)

    if epoch % 10 == 9:
        n_trj = 10
        avg_reward = 0
        for i in range(n_trj):
            tot_reward = 0
            obs, _ = env_sparse.reset()
            for t in range(50):
                current_state = torch.from_numpy(np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)).float().to(device)

                at, logprob, sigma = agent.get_action(current_state, test=True)
                next_obs, reward, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())
                tot_reward += reward

                done = terminated or truncated
                obs = next_obs
                if done:
                    break
            avg_reward += tot_reward
        avg_reward /= n_trj
        writer.add_scalar("Rewards/test_sparse_reward", avg_reward, epoch)

        reward_achieved = str(int(avg_reward))
        path_dir = "./saved_PPOs/" + environment_details["name"] + "/"
        exp_name = "R="+reward_achieved
        agent.save_model(path_dir, exp_name)




writer.close()



# env2 = gym.make(environment_details["gym_name"], render_mode='human')
# for i in range(10):
#     obs, _ = env2.reset()
#     for t in range(50):
#         env2.render()
#         current_state = torch.from_numpy(np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)).float().to(device)
#
#         at, logprob, sigma = agent.get_action(current_state)
#         next_obs, reward_to_goal, terminated, truncated, info = env2.step(at[0].detach().cpu().numpy())
#
#         done = terminated or truncated
#         obs = next_obs
#         if done:
#             break



























