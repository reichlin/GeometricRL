import numpy as np
import torch
import d3rlpy
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.manifold import Isomap
from torchvision.transforms import Resize, CenterCrop, Grayscale


def simulation(model, sim, device, exp_id, environment_details, render=False, n_episodes=10, use_images=0, image_rescale=False):
    if render:
        sim = gym.make(environment_details['gym_name'], continuing_task=False, render_mode='human') #max_episode_steps=max_T

    total_sparse_reward = 0
    total_dense_reward = 0
    for i in range(n_episodes):
        sparse_reward = 0
        dense_reward = 0
        obs, _ = sim.reset()

        if use_images == 1:
            grayscaler = Grayscale()
            cropper = CenterCrop(200)
            resizer = Resize(64)

            current_state = np.zeros((1, 4, 64, 64))
            obs, _ = sim.reset()
            for t in range(4):
                ot = sim.render()
                next_obs, reward_to_goal, terminated, truncated, info = sim.step(np.zeros(sim.action_space.shape[0]))
                image_torch = torch.from_numpy(np.transpose(ot, (2, 0, 1)).copy()).float() / 255.
                image_torch_resized = grayscaler(resizer(cropper(image_torch)))
                current_state[0, t] = image_torch_resized.detach().cpu().numpy()
        else:
            if exp_id < 3:
                current_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
            else:
                current_state = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)

        for t in range(1000):
            if render:
                sim.render()
            current_action = model.predict(current_state*255 if image_rescale else current_state)
            next_obs, reward, terminated, truncated, info = sim.step(np.squeeze(current_action))
            #total_sparse_reward += reward

            sparse_reward += reward #float(np.linalg.norm(next_obs['achieved_goal'] - next_obs['desired_goal']) <= 0.45)

            distance = np.linalg.norm(next_obs['achieved_goal'] - next_obs['desired_goal'], axis=-1)
            dense_reward += np.exp(-distance)

            obs = next_obs
            if use_images == 1:
                ot = sim.render()
                image_torch = torch.from_numpy(np.transpose(ot, (2, 0, 1)).copy()).float() / 255.
                image_torch_resized = grayscaler(resizer(cropper(image_torch)))
                current_state[0, :3] = current_state[0, 1:]
                current_state[0, 3] = image_torch_resized.detach().cpu().numpy()
            else:
                if exp_id < 3:
                    current_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
                else:
                    current_state = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
            if terminated or truncated:
                break

        # max_score = environment_details["d4rl_scores"][1]
        # min_score = environment_details["d4rl_scores"][0]
        # total_norm_reward += 100 * (norm_reward - min_score) / (max_score - min_score)
        total_sparse_reward += sparse_reward
        total_dense_reward += dense_reward

    return total_sparse_reward / n_episodes, total_dense_reward / n_episodes


def train_loop(agent, dataloader, env, writer, exp_name, environment_details, EPOCHS, args, device, policy_type, use_images=0):

    sparse_reward_records = []
    dense_norm_reward_records = []
    logs_idx = 0
    for epoch in tqdm(range(EPOCHS)):

        agent.train()
        for loader_idx, batch in enumerate(dataloader):
            log_losses = agent.update(batch=batch, epoch=epoch)

            writer.add_scalar("Losses/positive_loss", log_losses['L_pos'], logs_idx)
            writer.add_scalar("Losses/negative_loss", log_losses['L_neg'], logs_idx)
            writer.add_scalar("Losses/actor_loss", log_losses['L_pi'], logs_idx)
            if log_losses['L_trans'] is not None:
                writer.add_scalar("Losses/transition_loss", log_losses['L_trans'], logs_idx)
            logs_idx += 1

        if epoch % 1 == 0:
            agent.eval()
            avg_sparse_reward, avg_dense_score = simulation(agent, env, device, args.environment, environment_details, render=False, n_episodes=100, use_images=use_images)
            writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch)
            writer.add_scalar("Rewards/dense_score", avg_dense_score, epoch)

            sparse_reward_records.append(avg_sparse_reward)
            dense_norm_reward_records.append(avg_dense_score)
            np.savez("./saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))


def test_representation(env, exp_id, environment_details, agent, device, render=False):

    if render:
        env = gym.make(environment_details['gym_name'], continuing_task=False, render_mode='human')

    sparse_reward = 0
    dense_reward = 0

    all_a = []
    for i in np.linspace(-1, 1, 100):
        for j in np.linspace(-1, 1, 100):
            all_a.append(np.array([[i, j]]))
    all_a = torch.from_numpy(np.concatenate(all_a, 0)).float().to(device)

    tot_trj = 100
    for n_trj in range(tot_trj):

        states = []
        goals = []
        values = []

        obs, _ = env.reset()

        if exp_id < 3:
            current_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
        else:
            current_state = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)

        for t in range(1000):
            if render:
                env.render()

            torch_state = torch.from_numpy(current_state[:,:-2]).float().to(device).repeat(all_a.shape[0], 1)
            torch_next_states = agent.forward_dynamics(torch_state, all_a) #torch_state + agent.T(torch_state, all_a)

            current_values = agent.get_value(torch_next_states, torch.from_numpy(current_state[:,-2:]).float().to(device).repeat(torch_next_states.shape[0], 1))
            current_action = all_a[torch.argmax(current_values).detach().cpu().item()].detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(np.squeeze(current_action))
            states.append(current_state[:, :-2])
            goals.append(current_state[:, -2:])
            values.append(agent.get_value(torch.from_numpy(current_state[:, :-2]).float().to(device), torch.from_numpy(current_state[:, -2:]).float().to(device)).detach().cpu().item())
            sparse_reward += float(np.linalg.norm(next_obs['achieved_goal'] - next_obs['desired_goal']) <= 0.45)

            distance = np.linalg.norm(next_obs['achieved_goal'] - next_obs['desired_goal'], axis=-1)
            dense_reward += np.exp(-distance)

            obs = next_obs
            if exp_id < 3:
                current_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
            else:
                current_state = np.expand_dims(np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']), -1), 0)
            if terminated or truncated:
                break
        #print(n_trj, dense_reward, sparse_reward)
        # plt.plot(range(len(values)), values)
        # plt.show()

    return sparse_reward/tot_trj


def get_algo(algo_name, gamma, batch_size, device_flag, network_def, baseline_hyper, use_images=0):

    hidden_units = [network_def['hidden_units']] * network_def['n_layers']
    activation = 'relu' if network_def['activation'] == 0 else 'tanh'


    if use_images == 0:
        model_architecture = d3rlpy.models.VectorEncoderFactory(hidden_units=hidden_units,#[64, 64, 64, 64],
                                                                activation=activation,#'relu',
                                                                use_batch_norm=False,
                                                                dropout_rate=None)
    else:
        model_architecture = d3rlpy.models.PixelEncoderFactory(filters=[(64, 3, 1), (64, 3, 2), (64, 3, 2), (64, 3, 2)],
                                                               feature_size=64,
                                                               activation=activation,#'relu',
                                                               use_batch_norm=False,
                                                               dropout_rate=None)

    if algo_name == 'DDPG':
        algo_class = d3rlpy.algos.DDPG
        agent = d3rlpy.algos.DDPGConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'BC':
        algo_class = d3rlpy.algos.BC
        agent = d3rlpy.algos.BCConfig(batch_size=batch_size,
                                      observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                      policy_type='deterministic',
                                      encoder_factory=model_architecture).create(device=device_flag)
    elif algo_name == 'CQL':
        algo_class = d3rlpy.algos.CQL
        agent = d3rlpy.algos.CQLConfig(actor_encoder_factory=model_architecture,
                                       critic_encoder_factory=model_architecture,
                                       batch_size=batch_size,
                                       n_action_samples=baseline_hyper['n_actions'],
                                       gamma=gamma,
                                       observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                       conservative_weight=baseline_hyper['conservative_weight']).create(device=device_flag)
    elif algo_name == 'BCQ':
        algo_class = d3rlpy.algos.BCQ
        agent = d3rlpy.algos.BCQConfig(actor_encoder_factory=model_architecture,
                                       critic_encoder_factory=model_architecture,
                                       imitator_encoder_factory=model_architecture,
                                       batch_size=batch_size,
                                       n_action_samples=baseline_hyper['n_actions'],
                                       gamma=gamma,
                                       observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                       n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'BEAR':
        algo_class = d3rlpy.algos.BEAR
        agent = d3rlpy.algos.BEARConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        imitator_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_action_samples=baseline_hyper['n_actions'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'AWAC':
        algo_class = d3rlpy.algos.AWAC
        agent = d3rlpy.algos.AWACConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_action_samples=baseline_hyper['n_actions'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'PLAS':
        algo_class = d3rlpy.algos.PLAS
        agent = d3rlpy.algos.PLASConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        imitator_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_critics=baseline_hyper['n_critics'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        ).create(device=device_flag)
    elif algo_name == 'IQL':
        algo_class = d3rlpy.algos.IQL
        agent = d3rlpy.algos.IQLConfig(actor_encoder_factory=model_architecture,
                                       critic_encoder_factory=model_architecture,
                                       value_encoder_factory=model_architecture,
                                       batch_size=batch_size,
                                       gamma=gamma,
                                       observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                       n_critics=baseline_hyper['n_critics'],
                                       expectile=baseline_hyper['expectile']).create(device=device_flag)

    return algo_class, agent



