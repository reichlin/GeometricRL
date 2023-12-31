import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, CNN, Policy, Memory, fill_up_actions
import time
from torch.distributions.uniform import Uniform


class ContrastiveRL(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, network_def, var, offline_reg, device, use_images=0):
        super().__init__()

        self.lambd = offline_reg
        self.device = device
        self.dim_input = input_dim

        self.use_images = use_images
        if use_images == 0:
            self.phi = MLP(input_dim, a_dim, z_dim, 3, residual=True)
            self.psi = MLP(input_dim, 0, z_dim, 3, residual=True)

            self.pi = Policy(input_dim, goal_dim, a_dim, network_def, var)

            critic_params = list(self.phi.parameters()) + list(self.psi.parameters())
            self.opt_critic = torch.optim.Adam(critic_params, lr=1e-3)

            self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)
        else:
            self.phi = CNN(4, a_dim, z_dim)
            self.psi = CNN(4, 0, z_dim)

            self.pi_encoder = CNN(4, 0, z_dim)
            self.pi = Policy(z_dim, 0, a_dim, network_def, var)

            critic_params = list(self.phi.parameters()) + list(self.psi.parameters())
            self.opt_critic = torch.optim.Adam(critic_params, lr=1e-3)

            self.opt_actor = torch.optim.Adam(list(self.pi_encoder.parameters()) + list(self.pi.parameters()), lr=1e-3)



    def predict(self, s):
        if self.use_images == 0:
            st = torch.from_numpy(s[:, :self.dim_input]).float().to(self.device)
            gt = torch.from_numpy(s[:, self.dim_input:]).float().to(self.device)
        else:
            st = self.pi_encoder(torch.from_numpy(s).float().to(self.device))
            gt = None
        mu = self.pi.get_mean(st, gt)
        return torch.squeeze(mu).detach().cpu().numpy()

    def critic_loss(self, st, at, st1):

        z_sa = self.phi(st, at)
        z_g = self.psi(st1)

        logits = torch.einsum('ik, jk->ij', z_sa, z_g)
        L_pos = nn.functional.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]).to(self.device))

        return L_pos

    def train_actor(self, st, at, goal_t, ot=None, og=None, ot1=None):

        # TODO: get action space support and constrain policy to output mu in that range

        state = st if self.use_images == 0 else ot
        if self.use_images == 0:
            goal = st * 0
            goal[:, :goal_t.shape[-1]] = goal_t
        else:
            st = self.pi_encoder(ot)
            goal = og
            goal_t = None

        a = self.pi.sample_action(st, goal_t)
        z_sa = self.phi(state, a)
        z_g = self.psi(goal)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)

        log_prob_a_orig, _ = self.pi.get_log_prob(st, at, goal_t)

        tot_loss = (1 - self.lambd) * torch.mean(-1.0 * logits) - self.lambd * torch.mean(log_prob_a_orig)

        self.opt_actor.zero_grad()
        tot_loss.backward()
        self.opt_actor.step()

        return tot_loss

    def update(self, batch=None, batch_size=256, epoch=0):

        st, gt, at, _, _, st1, ot, og, ot1 = batch

        if self.use_images == 0:
            L_pos = self.critic_loss(st, at, st1)
        else:
            L_pos = self.critic_loss(ot, at, ot1)

        critic_loss = L_pos

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        L_pi = self.train_actor(st, at, gt, ot=ot, og=og, ot1=ot1)

        logs_losses = {'L_pos': L_pos.detach().cpu().item() if L_pos is not None else 0,
                       'L_neg': 0,
                       'L_trans': 0,
                       'L_pi': L_pi.detach().cpu().item() if L_pi is not None else 0,
                       'L_DQN': 0}

        return logs_losses

    def update_target(self, episode):
        return

