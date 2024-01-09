import os
import numpy as np
import matplotlib.pyplot as plt


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

z_dim = 32 # 32, 128, 512
reg = 1.0
batch_size = 256
use_images = 0

if use_images == 0:
    hypers = {1: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'DDPG'
              2: {'n_critics': 1, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BC'
              3: {'n_critics': 1, 'n_actions': 2, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'CQL'  1.0 10.0
              4: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BCQ'
              5: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BEAR'
              7: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'PLAS'
              8: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'IQL' 0.7 0.9
              9: {'n_critics': 1, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}} # 'ContrastiveRL'
    exploration_rewards = [34, 50, 78, 100, 126, 145, 176, 195]#[50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
    models_trained = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    seeds = [0, 1, 2]
else:
    hypers = {1: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'DDPG'
              2: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'BC'
              3: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'CQL'  1.0 10.0
              4: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'BCQ'
              5: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'BEAR'
              7: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'PLAS'
              8: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}, # 'IQL' 0.7 0.9
              9: {'n_critics': 2, 'n_actions': 2, 'conservative_weight': 5.0, 'expectile': 0.8}} # 'ContrastiveRL'
    exploration_rewards = [50, 40, 30, 20, 10, 1]
    models_trained = [0, 3, 4, 5, 7, 8, 9]
    seeds = [0]

environment = 2  # 0, 1, 2, 3, 4, 5
environment_details = experiments[environment]

all_results_sparse = np.ones((len(exploration_rewards), len(models_trained), len(seeds), 100)) * -201
all_results_dense = np.ones((len(exploration_rewards), len(models_trained), len(seeds), 100)) * -201 # env, expl, algo, seed, time

folder_name = "./saved_results/" + environment_details["name"] + "_img=0"
if use_images == 1:
    folder_name += "_img=1"
folder_name += "/"

for i, exploration in enumerate(exploration_rewards):
    for j, algo_id in enumerate(models_trained):
        for k, seed in enumerate(seeds):

            algo_name = algorithms[algo_id]

            if algo_id == 0:
                tmp_res_s = np.ones((3, 100)) * -201
                tmp_res_d = np.ones((3, 100)) * -201
                for z_id, z_dim in enumerate([32, 128, 512]):
                    exp_name = environment_details["name"] + "_R=-" + str(exploration)
                    exp_name += "_" + algo_name + "_BS=" + str(batch_size) + "_use_images=" + str(use_images)
                    exp_name += "_z_dim=" + str(z_dim)
                    exp_name += "_reg=" + str(reg)
                    exp_name += "_seed=" + str(seed)
                    if os.path.exists(folder_name + exp_name + ".npz"):
                        filenpz = np.load(folder_name + exp_name + ".npz")
                        sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                        tmp_res_s[z_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                        tmp_res_d[z_id, :sparse_reward_records.shape[0]] = dense_reward_records
                all_results_sparse[i, j, k] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                all_results_dense[i, j, k] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
            elif algo_id == 3 and use_images == 0:
                tmp_res_s = np.ones((2, 100)) * -201
                tmp_res_d = np.ones((2, 100)) * -201
                for w_id, c_w in enumerate([1.0, 10.0]):
                    exp_name = environment_details["name"] + "_R=-" + str(exploration)
                    exp_name += "_" + algo_name + "_BS=" + str(batch_size) + "_use_images=" + str(use_images)
                    baseline_hyper = {'n_critics': 1, 'n_actions': 2, 'conservative_weight': c_w, 'expectile': 0.9}
                    exp_name += "_baselines_hyper=" + str(baseline_hyper)
                    exp_name += "_seed=" + str(seed)
                    if os.path.exists(folder_name + exp_name + ".npz"):
                        filenpz = np.load(folder_name + exp_name + ".npz")
                        sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                        tmp_res_s[w_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                        tmp_res_d[w_id, :sparse_reward_records.shape[0]] = dense_reward_records
                all_results_sparse[i, j, k] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                all_results_dense[i, j, k] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
            elif algo_id == 8 and use_images == 0:
                tmp_res_s = np.ones((2, 100)) * -201
                tmp_res_d = np.ones((2, 100)) * -201
                for ex_id, exp in enumerate([0.7, 0.9]):
                    exp_name = environment_details["name"] + "_R=-" + str(exploration)
                    exp_name += "_" + algo_name + "_BS=" + str(batch_size) + "_use_images=" + str(use_images)
                    baseline_hyper = {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': exp}
                    exp_name += "_baselines_hyper=" + str(baseline_hyper)
                    exp_name += "_seed=" + str(seed)
                    if os.path.exists(folder_name + exp_name + ".npz"):
                        filenpz = np.load(folder_name + exp_name + ".npz")
                        sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                        tmp_res_s[ex_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                        tmp_res_d[ex_id, :sparse_reward_records.shape[0]] = dense_reward_records
                all_results_sparse[i, j, k] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                all_results_dense[i, j, k] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
            else:
                exp_name = environment_details["name"] + "_R=-" + str(exploration)
                exp_name += "_" + algo_name + "_BS=" + str(batch_size) + "_use_images=" + str(use_images)
                baseline_hyper = hypers[algo_id]
                exp_name += "_baselines_hyper=" + str(baseline_hyper)
                exp_name += "_seed=" + str(seed)
                if os.path.exists(folder_name + exp_name + ".npz"):
                    filenpz = np.load(folder_name + exp_name + ".npz")
                    sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                    all_results_sparse[i, j, k, :sparse_reward_records.shape[0]] = sparse_reward_records
                    all_results_dense[i, j, k, :sparse_reward_records.shape[0]] = dense_reward_records

print()

for j, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8, 9]):
    mu = np.mean(np.max(all_results_sparse[:, j, :], -1), -1)
    std = np.std(np.max(all_results_sparse[:, j, :], -1), -1)
    plt.plot(exploration_rewards, mu, label=algorithms[algo_id])
    plt.fill_between(exploration_rewards, mu-std, mu+std, alpha=0.2)
plt.legend()
plt.title("fetch Push")
plt.show()

for j, algo_id in enumerate(models_trained):
    mu = np.mean(np.max(all_results_sparse[:, j, :], -1), -1)
    std = np.std(np.max(all_results_sparse[:, j, :], -1), -1)
    plt.plot(np.array([-50, -40, -30, -20, -10, -1]), mu, label=algorithms[algo_id])
    plt.fill_between(np.array([-50, -40, -30, -20, -10, -1]), mu-std, mu+std, alpha=0.2)
plt.legend()
plt.title("fetch Reach imgs")
plt.show()

for j, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8, 9]):
    mu = np.mean(np.max(all_results_sparse[:, j, :], -1), -1)
    std = np.std(np.max(all_results_sparse[:, j, :], -1), -1)
    plt.plot(np.array([-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, -1]), mu, label=algorithms[algo_id])
    plt.fill_between(np.array([-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, -1]), mu-std, mu+std, alpha=0.2)
plt.legend()
plt.title("fetch Reach")
plt.show()

print()

# for k, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
#     np.savez("./saved_results/Mazes_" + algorithms[algo_id] + ".npz", all_results_sparse[:,:,k], all_results_dense[:,:,k])

# for i, env in enumerate([0, 1, 2, 3, 4, 5]):
#     for j, expl in enumerate([0, 1, 2]):
#         for k, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
#             y = np.zeros(100)
#             for t in range(100):
#                 y[t] = np.mean(all_results[i, j, k, 0, max(0, t-2):min(99, t+2)])
#             plt.plot(range(100), y, label=algorithms[algo_id])
#         plt.title(experiments[env]["name"] + " " + exploration_processes[expl])
#         plt.legend()
#         plt.show()


# for i, env in enumerate([0, 1, 2]):
#     for j, expl in enumerate([0, 1, 2]):
#         print(experiments[env]["name"] + " " + exploration_processes[expl])
#         for k, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
#             if np.max(all_results[i, j, k, :, :]) < 0:
#                 print(" - ")
#             else:
#                 max_rewards = np.max(all_results[i, j, k, :, :], -1)
#                 max_rewards = max_rewards[np.nonzero(np.maximum(0, max_rewards+0.5))]
#                 mu = np.mean(max_rewards)
#                 sigma = np.std(max_rewards)
#                 print("& $"+f'{mu:.2f}'+"$ \scriptsize{$\pm "+f'{sigma:.2f}'+"$}", end=" ")
#         print()
#     print("-----------------------------------------")
#
# print()

# for environment in [0, 1, 2, 3, 4, 5]:
#     for exploration in [0, 1, 2]:
#         for z_dim in [512]:
#             for seed in [0]:
#
#                 algo_name = algorithms[algo_id]
#                 exploration_strategy = exploration_processes[exploration]
#                 environment_details = experiments[environment]
#
#                 exp_name = environment_details["name"] + "_" + exploration_strategy
#                 exp_name += "_" + algo_name + "_BS=" + str(batch_size)
#                 exp_name += "_z_dim=" + str(z_dim)
#                 exp_name += "_reg=" + str(reg)
#                 exp_name += "_seed=" + str(0)
#
#                 if os.path.exists("./saved_results/" + exp_name + ".npz"):
#
#                     filenpz = np.load("./saved_results/" + exp_name + ".npz")
#                     sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
#
#                     print(environment_details["name"], exploration_strategy, z_dim, ": ", np.max(sparse_reward_records))
#
# print()
#
# plt.plot(range(sparse_reward_records.shape[0]), sparse_reward_records)
# plt.show()
#
# plt.plot(range(dense_reward_records.shape[0]), dense_reward_records)
# plt.show()












