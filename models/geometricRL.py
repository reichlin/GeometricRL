import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, CNN, Policy, Memory, fill_up_actions
import time
from torch.distributions.uniform import Uniform


class GeometricRL(nn.Module):

    def __init__(self, dim_input, dim_goal, dim_z, dim_a, gamma, memory_size=0, all_actions_idx=None, all_actions=None, network_def=None, K=2, var=0.1, R_gamma=1., reg=1.0, policy_type=0, policy_clip=None, device=None, use_images=0):
        super().__init__()

        self.replay_buffer = Memory(memory_size)
        self.all_actions_idx = all_actions_idx
        self.all_actions = all_actions

        self.policy_clip = policy_clip
        self.reg = reg

        self.device = device

        self.dim_input = dim_input
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.gamma = gamma

        self.policy_type = policy_type

        self.R_gamma = R_gamma

        self.use_images = use_images
        if use_images == 0:
            self.phi = MLP(dim_input, 0, dim_z, 3, residual=True)
            self.policy_type = policy_type
            if policy_type == -1:
                self.T = MLP(dim_input, dim_a, dim_input, 2)  # dim_input +
                critic_params = list(self.phi.parameters()) + list(self.T.parameters())
            else:
                critic_params = self.phi.parameters()
                self.pi = Policy(dim_input, dim_goal, dim_a, network_def, var)
                self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)
        else:
            self.phi = CNN(4, 0, dim_z)
            self.policy_type = policy_type

            critic_params = self.phi.parameters()
            self.pi_encoder = CNN(4, 0, dim_z)
            self.pi = Policy(dim_z, 0, dim_a, network_def, var)
            self.opt_actor = torch.optim.Adam(list(self.pi_encoder.parameters()) + list(self.pi.parameters()), lr=1e-3)



        self.opt_critic = torch.optim.Adam(critic_params, lr=1e-3)


    def get_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def predict(self, s):
        if self.use_images == 0:
            st = torch.from_numpy(s[:, :self.dim_input]).float().to(self.device)
            gt = torch.from_numpy(s[:, self.dim_input:]).float().to(self.device)
        else:
            st = self.pi_encoder(torch.from_numpy(s).float().to(self.device))
            gt = None

        if self.policy_type == 0:
            mu = self.pi(st, gt)
        else:
            mu = self.pi.get_mean(st, gt)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_action(self, state, current_obst, goal, test=False):

        if not test and np.random.rand() < self.epsilon:
            a_idx = np.random.randint(0, self.all_actions.shape[0])
        else:
            zg = self.phi(goal)
            all_next_s = state.repeat(self.dim_a, 1) + self.T(state.repeat(self.dim_a, 1), torch.cat([torch.from_numpy(self.all_actions).view(-1, self.dim_input).float().to(self.device), current_obst.repeat(self.dim_a, 1)], -1))
            next_states_dist = self.get_distance(self.phi(all_next_s), zg)
            a_idx = torch.argmin(next_states_dist).detach().cpu().item()
        return a_idx

    def get_value(self, s, s_g):
        #goal = torch.cat([s_g, s[:, s_g.shape[1]:]], -1)
        # goal = s_g.repeat(1, 2)
        # goal[:, 2:] *= 0
        if self.use_images == 0:
            goal = s * 0
            goal[:, :s_g.shape[-1]] = s_g
        else:
            goal = s_g

        z = self.phi(s)
        z_g = self.phi(goal)
        dist = self.get_distance(z, z_g)
        neg_dist = - self.get_distance(z, z_g)  # TODO: minus -> I want to maximize V
        return neg_dist
        # V = (self.gamma ** dist) * 1
        # return V

    def critic_loss(self, st, at, st1):

        z = self.phi(st)
        z1 = self.phi(st1)

        action_distance = 1
        L_pos = torch.mean((self.get_distance(z1, z) - action_distance) ** 2)

        idx = np.zeros((z.shape[0], z.shape[0] - 1))
        for i in range(z.shape[0]):
            idx[i] = np.delete(np.arange(z.shape[0]), i)
        z1_rep = torch.cat([torch.unsqueeze(z1[i], 0) for i in idx], 0)
        dist_z_perm = - torch.log(torch.cdist(torch.unsqueeze(z, 1), z1_rep) + 1e-6)
        L_neg = torch.mean(dist_z_perm)

        return L_pos, L_neg

    def train_actor(self, st, gt, st1, at, ot=None, og=None, ot1=None):

        # TODO: get action space support and constrain policy to output mu in that range

        if self.policy_type == 0:

            at = self.pi(st, gt)
            s1_pred = self.T(st, at)
            V_next = self.get_value(s1_pred, gt)
            tot_loss = - torch.mean(V_next)


        elif self.policy_type == 1:
            state = st if self.use_images == 0 else ot
            next_state = st1 if self.use_images == 0 else ot1
            goal = gt if self.use_images == 0 else og
            if self.use_images == 1:
                st = self.pi_encoder(ot)
                gt = None
            log_prob, entropy = self.pi.get_log_prob(st, at, gt)
            Vt = self.get_value(state, goal)
            Vt1 = self.get_value(next_state, goal)
            Adv = (Vt1 - Vt).detach() #  * self.gamma
            Adv_skew = Adv #(torch.exp( Adv * self.R_gamma)-1) #torch.clamp(Adv, -1, 1) # (torch.exp(  * self.R_gamma)-1) / 10.
            tot_loss = - torch.mean(Adv_skew * log_prob)

        self.opt_actor.zero_grad()
        tot_loss.backward()
        if self.policy_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.policy_clip)
        self.opt_actor.step()

        return tot_loss

    def forward_dynamics(self, s, a):
        return s + self.T(s, a)

    def update(self, batch=None, batch_size=256, epoch=0):

        if batch is None:
            data = self.replay_buffer.sample(batch_size)

            st = torch.cat([x['st'] for x in data], 0)
            obst = torch.cat([x['obst'] for x in data], 0)
            gt = torch.cat([x['goal_t'] for x in data], 0)
            at = torch.cat([torch.from_numpy(x['a']).float().view(1, -1).to(st.device) for x in data], 0)
            a_idx = np.array([x['a_idx'] for x in data])
            st1 = torch.cat([x['st1'] for x in data], 0)
        else:
            st, gt, at, _, _, st1, ot, og, ot1 = batch

        L_pos, L_neg, L_trans, L_pi = None, None, None, None
        if self.policy_type == -1:
            L_pos, L_neg = self.critic_loss(st, at, st1)

            s1_hat = self.forward_dynamics(st, at)
            L_trans = torch.mean(torch.sum((st1 - s1_hat)**2, -1))

            critic_loss = L_pos + self.reg * L_neg + L_trans

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
        else:
            if self.use_images == 0:
                L_pos, L_neg = self.critic_loss(st, at, st1)
            else:
                L_pos, L_neg = self.critic_loss(ot, at, ot1)
            critic_loss = L_pos + self.reg * L_neg

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

            L_pi = self.train_actor(st, gt, st1, at, ot=ot, og=og, ot1=ot1)

        logs_losses = {'L_pos': L_pos.detach().cpu().item() if L_pos is not None else 0,
                       'L_neg': L_neg.detach().cpu().item() if L_neg is not None else 0,
                       'L_trans': L_trans.detach().cpu().item() if L_trans is not None else 0,
                       'L_pi': L_pi.detach().cpu().item() if L_pi is not None else 0,
                       'L_DQN': 0}

        return logs_losses

    def update_target(self, episode):
        return

