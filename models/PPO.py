import torch
from torch import optim, nn
from my_utils.model_utils import ActorCritic


class BatchData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminal = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminal.clear()


def calc_rtg(rewards, is_terminals, gamma):
    # Calculates reward-to-go
    assert len(rewards) == len(is_terminals)
    rtgs = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        rtgs.insert(0, discounted_reward)
    return rtgs


class PPO:

    def __init__(self, input_dim, a_dim, a_max, device):

        self.action_dim = a_dim
        self.batchdata = BatchData()
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.001
        self.eps_clip = 0.2
        self.gamma = 0.9
        self.c1 = 1.  # VF loss coefficient
        self.c2 = 0.01 #0.01 # Entropy bonus coefficient [0.01 for fetch_Reach]
        self.c2_schedule = 1.
        self.K_epochs = 5  # num epochs to train on batch data
        #self.epsilon = 0.9

        self.policy = ActorCritic(input_dim, a_dim, a_max).to(device)
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = optim.RMSprop(self.policy.parameters(), self.lr)

        self.old_policy = ActorCritic(input_dim, a_dim, a_max).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, st, test=False):
        a, logprob, sigma = self.old_policy.get_action(st, test=test)
        return a, logprob, sigma

    def update(self):
        """
            Updates the actor-critic networks for current batch data
        """
        rtgs = torch.tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma)).float().to(self.device)
        #rtgs /= torch.abs(torch.mean(rtgs))

        old_states = torch.cat([x for x in self.batchdata.states], 0).to(self.device)
        old_actions = torch.cat([x for x in self.batchdata.actions], 0).to(self.device)
        old_logprobs = torch.cat([x for x in self.batchdata.logprobs], 0).to(self.device)


        for _ in range(self.K_epochs):
            logprobs, state_vals, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())  # new probs over old probs

            # Calc advantages
            A = rtgs - state_vals.detach()  # old rewards and old states evaluated by curr policy

            # Actor loss using CLIP loss
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # minus to maximize

            # Critic loss fitting to reward-to-go with entropy bonus
            critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals) - self.c2 * torch.mean(dist_entropy)

            loss = actor_loss + critic_loss

            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()

        # Replace old policy with new policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.c2 *= self.c2_schedule

        return self.MSE_loss(rtgs, state_vals).detach().cpu().item(), torch.mean(dist_entropy).detach().cpu().item()

    def push_batchdata(self, st, a, logprob, r, done):
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob)
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)

    def clear_batchdata(self):
        self.batchdata.clear()

    def save_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        torch.save(self.policy.state_dict(), fname_psi)

    def load_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        state_dict_psi = torch.load(fname_psi, map_location=self.device)
        self.policy.load_state_dict(state_dict_psi)
        self.old_policy.load_state_dict(state_dict_psi)
        self.policy.eval()
        self.old_policy.eval()
