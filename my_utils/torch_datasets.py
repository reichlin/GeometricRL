import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.distributions.geometric import Geometric


class Dataset_Uniform_precollected(Dataset):

    def __init__(self, states, actions, rewards, terminations, next_states, device, size, obs_shape, images=None, images_goal=None):

        self.device = device

        # TODO: repeated images for dynamics

        '''
            rewards is always zero, last step before termination is discarded
        '''

        self.states = torch.from_numpy(states).float()[:, :obs_shape].to(self.device)
        if images is not None:
            self.images = torch.from_numpy(images).float().to(self.device)
            self.images_goal = torch.from_numpy(images_goal).float().to(self.device)
        else:
            self.images = None
            self.images_goal = None
        self.goals = torch.from_numpy(states).float()[:, obs_shape:].to(self.device)

        # goals = states[:, obs_shape:]
        # goal_idx = np.zeros(states.shape[0])
        # for i in tqdm(range(states.shape[0])):
        #     min_d = 10000
        #     idx = -1
        #     for j in range(states.shape[0]):
        #         d_j = np.linalg.norm(goals[i] - states[j, :goals.shape[-1]])
        #         if np.linalg.norm(states[i, obs_shape:] - states[j, obs_shape:]) < 0.02:
        #             if min_d > d_j:
        #                 min_d = d_j
        #                 idx = j
        #     goal_idx[i] = idx

        # goals = states[:, obs_shape:]
        # self.goals = self.states * 0
        # self.goals[:, :goals.shape[-1]] = goals

        self.actions = torch.from_numpy(actions).float().to(self.device)
        self.rewards = torch.from_numpy(rewards).float().to(self.device)
        self.terminations = torch.from_numpy(terminations).float().to(self.device)
        self.next_states = torch.from_numpy(next_states).float()[:, :obs_shape].to(self.device)

        self.len = self.states.shape[0]
        self.size = size

    def __len__(self):
        # return self.len
        return self.size

    def __getitem__(self, idx):

        idx = np.random.randint(0, self.len)

        s = self.states[idx]#.to(self.device)
        o = self.images[idx] if self.images is not None else self.states[idx]*0
        og = self.images_goal[idx] if self.images_goal is not None else self.states[idx]*0
        g = self.goals[idx]#.to(self.device)
        a = self.actions[idx]#.to(self.device)
        r = self.rewards[idx]#.to(self.device)
        term = self.terminations[idx]#.to(self.device)
        s1 = self.next_states[idx]#.to(self.device)

        if term == 0:
            o1 = self.images[idx + 1] if self.images is not None else self.states[idx]*0
        else:
            o1 = self.images[idx] if self.images is not None else self.states[idx]*0

        return s, g, a, r, term, s1, o, og, o1


class Dataset_Minari(Dataset):
    def __init__(self, dataset, device, size, include_achiev_goal=False):

        self.dataset = dataset
        self.len = dataset.total_steps
        self.device = device

        states = None
        goals = None
        actions = None
        rewards = None
        term = None
        next_states = None
        for e in tqdm(self.dataset.episode_indices):
            trj = self.dataset[e]
            if trj.observations['observation'].shape[0] < 2 or trj.actions.shape[0] < 1:
                print("empty trj", e)
            elif (trj.observations['observation'].shape[0] - 1) == trj.actions.shape[0]:
                if include_achiev_goal:
                    st = np.concatenate((trj.observations['achieved_goal'][:-1], trj.observations['observation'][:-1]), -1)
                    states = st if states is None else np.concatenate((states, st), 0)
                else:
                    states = trj.observations['observation'][:-1] if states is None else np.concatenate((states, trj.observations['observation'][:-1]), 0)
                goals = trj.observations['desired_goal'][:-1] if goals is None else np.concatenate((goals, trj.observations['desired_goal'][:-1]), 0)
                actions = trj.actions if actions is None else np.concatenate((actions, trj.actions), 0)
                rewards = trj.rewards if rewards is None else np.concatenate((rewards, trj.rewards), 0)
                term = trj.terminations*1. if term is None else np.concatenate((term, trj.terminations*1.), 0)
                if include_achiev_goal:
                    st1 = np.concatenate((trj.observations['achieved_goal'][1:], trj.observations['observation'][1:]), -1)
                    next_states = st1 if next_states is None else np.concatenate((next_states, st1), 0)
                else:
                    next_states = trj.observations['observation'][1:] if next_states is None else np.concatenate((next_states, trj.observations['observation'][1:]), 0)
            else:
                print("non complete trj", e)

        self.states = torch.from_numpy(states).float().to(self.device)
        self.goals = torch.from_numpy(goals).float().to(self.device)

        # self.goals = self.states * 0
        # self.goals[:, :goals.shape[-1]] = goals

        self.actions = torch.from_numpy(actions).float().to(self.device)
        self.rewards = torch.from_numpy(rewards).float().to(self.device)
        self.term = torch.from_numpy(term).float().to(self.device)
        self.next_states = torch.from_numpy(next_states).float().to(self.device)

        self.len = self.states.shape[0]
        self.size = size

    def __len__(self):
        # return self.len
        return self.size

    def __getitem__(self, idx):

        idx = np.random.randint(0, self.len)

        s = self.states[idx]
        g = self.goals[idx]
        a = self.actions[idx]
        r = self.rewards[idx]
        term = self.term[idx]
        s1 = self.next_states[idx]

        return s, g, a, r, term, s1


class Dataset_Visitation(Dataset):

    def __init__(self, dataset_tuple, offline_dataset, device, size, gamma, obs_shape=0, include_achiev_goal=False, images=None, images_goal=None):


        self.device = device
        self.GEOM = Geometric(1-gamma)

        if offline_dataset is None:
            states, actions, rewards, terminations, next_states = dataset_tuple

            trj_idx = np.nonzero(terminations[:, 0])[0]
            i = 0
            self.states, self.goals, self.actions, self.rewards, self.terminations, self.next_states, self.images, self.images_goal = [], [], [], [], [], [], [], []
            for j in trj_idx:
                self.states.append(torch.from_numpy(states[i:j, :obs_shape]).float().to(self.device))
                self.goals.append(torch.from_numpy(states[i:j, obs_shape:]).float().to(self.device))
                # goal_states = states[i:j, :obs_shape] * 0
                # goal_states[:, :-obs_shape] = states[i:j, obs_shape:]
                # self.goals.append(torch.from_numpy(goal_states).float().to(self.device))

                if images is not None:
                    self.images.append(torch.from_numpy(images[i:j]).float().to(self.device))
                    self.images_goal.append(torch.from_numpy(images_goal[i:j]).float().to(self.device))
                else:
                    self.images = None
                    self.images_goal = None

                self.actions.append(torch.from_numpy(actions[i:j]).float().to(self.device))
                self.rewards.append(torch.from_numpy(rewards[i:j]).float().to(self.device))
                self.terminations.append(torch.from_numpy(terminations[i:j]).float().to(self.device))
                self.next_states.append(torch.from_numpy(states[i + 1:j + 1, :obs_shape]).float().to(self.device))
                i = j + 1

        else:
            self.states, self.goals, self.actions, self.rewards, self.terminations, self.next_states = [], [], [], [], [], []
            for e in tqdm(offline_dataset.episode_indices):
                states, goals, actions, rewards, term, next_states = None, None, None, None, None, None
                trj = offline_dataset[e]
                if trj.observations['observation'].shape[0] < 2 or trj.actions.shape[0] < 1:
                    print("empty trj", e)
                elif (trj.observations['observation'].shape[0] - 1) == trj.actions.shape[0]:
                    if include_achiev_goal:
                        st = np.concatenate((trj.observations['achieved_goal'][:-1], trj.observations['observation'][:-1]), -1)
                        states = st if states is None else np.concatenate((states, st), 0)
                    else:
                        states = trj.observations['observation'][:-1] if states is None else np.concatenate((states, trj.observations['observation'][:-1]), 0)
                    goals = trj.observations['desired_goal'][:-1] if goals is None else np.concatenate((goals, trj.observations['desired_goal'][:-1]), 0)
                    actions = trj.actions if actions is None else np.concatenate((actions, trj.actions), 0)
                    rewards = trj.rewards if rewards is None else np.concatenate((rewards, trj.rewards), 0)
                    term = trj.terminations * 1. if term is None else np.concatenate((term, trj.terminations * 1.), 0)
                    if include_achiev_goal:
                        st1 = np.concatenate((trj.observations['achieved_goal'][1:], trj.observations['observation'][1:]), -1)
                        next_states = st1 if next_states is None else np.concatenate((next_states, st1), 0)
                    else:
                        next_states = trj.observations['observation'][1:] if next_states is None else np.concatenate((next_states, trj.observations['observation'][1:]), 0)
                else:
                    print("non complete trj", e)
                if states is not None:
                    self.states.append(torch.from_numpy(states).float().to(self.device))
                    self.goals.append(torch.from_numpy(goals).float().to(self.device))
                    # goal_states = states * 0
                    # goal_states[:, :goals.shape[-1]] = goals
                    # self.goals.append(torch.from_numpy(goal_states).float().to(self.device))
                    self.actions.append(torch.from_numpy(actions).float().to(self.device))
                    self.rewards.append(torch.from_numpy(rewards).float().to(self.device))
                    self.terminations.append(torch.from_numpy(term).float().to(self.device))
                    self.next_states.append(torch.from_numpy(next_states).float().to(self.device))

        self.n_trj = len(self.states)
        self.size = size

    def __len__(self):
        # return self.len
        return self.size

    def __getitem__(self, idx):

        trj_i = np.random.randint(0, self.n_trj)

        len_trj = self.states[trj_i].shape[0]

        t = int(torch.clamp(self.GEOM.sample(), max=len_trj - 1).item())
        i = np.random.randint(0, len_trj-t)

        s = self.states[trj_i][i]
        o = self.images[trj_i][i] if self.images is not None else self.states[trj_i][i]*0
        og = self.images_goal[trj_i][i] if self.images_goal is not None else self.states[trj_i][i]*0
        g = self.goals[trj_i][i]
        a = self.actions[trj_i][i]
        r = self.rewards[trj_i][i]
        term = self.terminations[trj_i][i]
        s1 = self.next_states[trj_i][i+t]

        if term == 0:
            o1 = self.images[trj_i][i+t] if self.images is not None else self.states[trj_i][i]*0
        else:
            o1 = self.images[trj_i][i] if self.images is not None else self.states[trj_i][i]*0

        return s, g, a, r, term, s1, o, og, o1








