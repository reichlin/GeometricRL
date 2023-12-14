import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, Policy
import time
from torch.distributions.uniform import Uniform


class QuasiMetric(nn.Module):

    def __init__(self, dim_input, dim_goal, dim_z, dim_a, R_gamma, network_def=None, var=0.1, device=None):
        super().__init__()

        self.device = device

        self.dim_input = dim_input
        self.dim_z = dim_z
        self.dim_a = dim_a

        self.R_gamma = R_gamma

        self.f = MLP(dim_input, 0, dim_z, 4)
        self.d = MLP(2*dim_z, 0, 1, 4)
        self.T = MLP(dim_z+dim_a, 0, dim_z, 4)
        self.pi = Policy(dim_input, dim_goal, dim_a, network_def, var)

        self.opt_critic = torch.optim.Adam(list(self.f.parameters())+list(self.d.parameters())+list(self.T.parameters()), lr=1e-3)
        self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)

    def get_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def predict(self, s):
        st = torch.from_numpy(s[:, :self.dim_input]).float().to(self.device)
        gt = torch.from_numpy(s[:, self.dim_input:]).float().to(self.device)
        mu = self.pi.get_mean(st, gt)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_value(self, s, s_g):
        goal = torch.cat([s_g, s[:, s_g.shape[1]:]], -1)
        z = self.f(s)
        z_g = self.f(goal)
        neg_dist = - self.get_distance(z, z_g)
        return neg_dist

    def critic_loss(self, st, sg, at, rt, st1):

        zt = self.f(st)
        zg = self.f(sg)
        zt1 = self.f(st1)
        zt1_hat = self.T(torch.cat([zt, at], -1))

        reward = rt - 1

        epsilon = 0.25
        lamb = 0.01
        softplus = nn.Softplus()
        relu = nn.ReLU()
        dist_goal = torch.mean(-100 * softplus(5 - self.d(torch.cat([zt, zg], -1))))
        dist_states = torch.mean(relu(self.d(torch.cat([zt, zt1], -1)) + reward)**2) - epsilon

        L_qm = - dist_goal + lamb * dist_states

        L_trans = torch.mean(self.get_distance(zt1, zt1_hat)**2)

        return L_qm, L_trans

    def actor_loss(self, st, goal_t, st1, at):

        log_prob = self.pi.get_log_prob(st, at, goal_t)
        Vt = self.get_value(st, goal_t)
        Vt1 = self.get_value(st1, goal_t)
        Adv = (Vt1 - Vt).detach()
        Adv_skew = torch.exp(Adv * self.R_gamma)-1
        tot_loss = - torch.mean(Adv_skew * log_prob)

        self.opt_actor.zero_grad()
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt_actor.step()

        return tot_loss

    def fit(self, batch):

        st, goal_t, at, rt, _, st1 = batch
        sg = torch.cat([goal_t, st[:, goal_t.shape[1]:]], -1)

        L_qm, L_trans = self.critic_loss(st, sg, at, rt, st1)

        critic_loss = L_qm + L_trans

        self.opt_critic.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt_critic.step()

        L_pi = self.actor_loss(st, goal_t, st1, at)

        return L_qm.detach().cpu().item(), L_trans.detach().cpu().item(), L_pi.detach().cpu().item()

