import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.transforms import Resize, CenterCrop, Grayscale
from PIL import Image
import argparse

from models.PPO import PPO


# name Minari, name Gym, state space dim, goal space dim, action space dim
experiments = {0: {"name": "fetch_Reach", "gym_name": 'FetchReachDense-v2', "sim_name": 'FetchReach-v2', "state_dim": 10, "goal_dim": 3, "action_dim": 4},
               1: {"name": "fetch_PickandPlace", "gym_name": 'FetchPickAndPlaceDense-v2', "sim_name": 'FetchPickAndPlace-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4},
               2: {"name": "fetch_Push", "gym_name": 'FetchPushDense-v2', "sim_name": 'FetchPush-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4}}

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--environment', default=2, type=int)
parser.add_argument('--reward', default=50, type=int)

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

env_sparse = gym.make(environment_details["sim_name"], render_mode='rgb_array', max_episode_steps=104) #, render_mode='human')


agent = PPO(input_dim+goal_dim, a_dim, 1, device)

agent_optimal = PPO(input_dim+goal_dim, a_dim, 1, device)
reward_achieved = "-" + str(20)
path_dir = "./saved_PPOs/" + environment_details["name"] + "/"
exp_name = "R=" + reward_achieved
agent_optimal.load_model(path_dir, exp_name)

reward = args.reward

for reward in [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]: #range(1, 51):
    print(reward, end=" ")

    reward_achieved = "-"+str(reward)
    path_dir = "./saved_PPOs/" + environment_details["name"] + "/"
    exp_name = "R=" + reward_achieved
    agent.load_model(path_dir, exp_name)

    EPOCHS = 100

    grayscaler = Grayscale()
    cropper = CenterCrop(200)
    resizer = Resize(64)

    states, images, images_goal, actions, rewards, terminations, next_states = None, None, None, None, None, None, None
    for epoch in tqdm(range(EPOCHS)):
        tmp_states, tmp_images, tmp_images_goal, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], [], [], []

        past_images = np.zeros((4, 64, 64))
        current_images = np.zeros((4, 64, 64))

        obs, _ = env_sparse.reset()
        for t in range(4):
            ot = env_sparse.render()
            next_obs, reward_to_goal, terminated, truncated, info = env_sparse.step(np.zeros(a_dim))
            image_torch = torch.from_numpy(np.transpose(ot, (2, 0, 1)).copy()).float()/255.
            image_torch_resized = grayscaler(resizer(cropper(image_torch)))
            past_images[t] = image_torch_resized.detach().cpu().numpy()

        for t in range(50):
            ot = env_sparse.render()
            st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)

            at, logprob, sigma = agent.get_action(torch.from_numpy(st).float().to(device), test=True)  # TODO test=True ???
            next_obs, reward_to_goal, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())

            done = terminated or truncated

            tmp_states.append(st)
            #tmp_images.append(np.expand_dims(ot, 0))
            image_torch = torch.from_numpy(np.transpose(ot, (2, 0, 1)).copy()).float()/255.
            image_torch_resized = grayscaler(resizer(cropper(image_torch)))
            current_images[:3] = past_images[1:]
            current_images[3] = image_torch_resized.detach().cpu().numpy()
            tmp_images.append(np.expand_dims(current_images.copy(), 0))
            past_images = current_images.copy()

            tmp_actions.append(at.detach().cpu().numpy())
            tmp_rewards.append(np.array([[reward_to_goal]]))
            tmp_terminations.append(np.array([[done * 1.]]))
            tmp_next_states.append(np.expand_dims(np.concatenate((next_obs['observation'], next_obs['desired_goal']), -1), 0))

            obs = next_obs

        tmp_terminations[-1][0, 0] = 1

        for t in range(50):
            ot = env_sparse.render()
            st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
            at, logprob, sigma = agent_optimal.get_action(torch.from_numpy(st).float().to(device), test=True)
            next_obs, reward_to_goal, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())

            if reward_to_goal > -0.5:
                break

        image_torch = torch.from_numpy(np.transpose(ot, (2, 0, 1)).copy()).float() / 255.
        image_torch_resized = grayscaler(resizer(cropper(image_torch)))
        goal_image_single = np.expand_dims(image_torch_resized.detach().cpu().numpy(), 0)
        # tmp_images_goal = np.tile(goal_image_single, (50, 4, 1, 1))

        if states is None:
            states = np.concatenate(tmp_states, 0)
            # images_torch = torch.from_numpy(np.transpose(np.concatenate(tmp_images, 0), (0, 3, 1, 2))).float()/255.
            # images_torch_resized = grayscaler(resizer(cropper(images_torch)))
            # images = images_torch_resized.detach().cpu().numpy()
            images = np.concatenate(tmp_images, 0)
            images_goal = np.tile(goal_image_single, (50, 4, 1, 1))
            actions = np.concatenate(tmp_actions, 0)
            rewards = np.concatenate(tmp_rewards, 0)
            terminations = np.concatenate(tmp_terminations, 0)
            next_states = np.concatenate(tmp_next_states, 0)
        else:
            states = np.concatenate((states, np.concatenate(tmp_states, 0)), 0)
            # images_torch = torch.from_numpy(np.transpose(np.concatenate(tmp_images, 0), (0, 3, 1, 2))).float()/255.
            # images_torch_resized = grayscaler(resizer(cropper(images_torch)))
            # images = np.concatenate((images, images_torch_resized.detach().cpu().numpy()), 0)
            images = np.concatenate((images, np.concatenate(tmp_images, 0)), 0)
            images_goal = np.concatenate((images_goal, np.tile(goal_image_single, (50, 4, 1, 1))), 0)
            actions = np.concatenate((actions, np.concatenate(tmp_actions, 0)), 0)
            rewards = np.concatenate((rewards, np.concatenate(tmp_rewards, 0)), 0)
            terminations = np.concatenate((terminations, np.concatenate(tmp_terminations, 0)), 0)
            next_states = np.concatenate((next_states, np.concatenate(tmp_next_states, 0)), 0)

    file_name = "./datasets/" + environment_details["name"] + "/" + exp_name + ".npz"
    np.savez(file_name, states, images, images_goal, actions, rewards, terminations, next_states)


# tot_rewards = []
# for reward in range(50, 0, -1):
#     exp_name = "R=-"+str(reward)
#     file_name = "./datasets_var/" + environment_details["name"] + "/" + exp_name + ".npz"
#     filenpz = np.load(file_name)
#     states, images, images_goal, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5'], filenpz['arr_6']
#     tot_rewards.append(np.sum(rewards > -0.5))
# plt.plot(range(1, 51), tot_rewards)
# plt.show()

tot_rewards = []
for reward in [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
    exp_name = "R=-"+str(reward)
    file_name = "./datasets/" + environment_details["name"] + "/" + exp_name + ".npz"
    filenpz = np.load(file_name)
    states, images, images_goal, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5'], filenpz['arr_6']
    tot_rewards.append(np.sum(rewards > -0.5)/100)
plt.plot([35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], tot_rewards)
plt.show()

print()























