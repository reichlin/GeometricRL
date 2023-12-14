import os
import numpy as np
import matplotlib.pyplot as plt


exploration_processes = {0: "Random",
                         1: "Ornstein-Uhlenbeck",
                         2: "Minari"}

# name Minari, name Gym, state space dim, goal space dim, action space dim
experiments = {0: {"name": "point_uMaze", "minari_name": 'pointmaze-umaze-v1', "gym_name": 'PointMaze_UMaze-v3', "d4rl_scores": [23.85, 161.86], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               1: {"name": "point_Medium", "minari_name": 'pointmaze-medium-v1', "gym_name": 'PointMaze_Medium-v3', "d4rl_scores": [13.13, 277.39], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               2: {"name": "point_Large", "minari_name": 'pointmaze-large-v1', "gym_name": 'PointMaze_Large-v3', "d4rl_scores": [6.7, 273.99], "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               3: {"name": "ant_uMaze", "minari_name": 'antmaze-umaze-diverse-v0', "gym_name": 'AntMaze_UMaze-v3', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8},
               4: {"name": "ant_Medium", "minari_name": 'antmaze-medium-diverse-v0', "gym_name": 'AntMaze_Medium_Diverse_GR-v3', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8},
               5: {"name": "ant_Large", "minari_name": 'antmaze-large-diverse-v0', "gym_name": 'AntMaze_Large_Diverse_GR-v3', "d4rl_scores": [0.0, 1.0], "state_dim": 29, "goal_dim": 2, "action_dim": 8}}

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
              10: 'QuasiMetric'
              }


environment = 0  # 0, 1, 2, 3, 4, 5
exploration = 0  # 0, 1, 2
algo_id = 0
z_dim = 32 # 32, 128, 512
reg = 1.0
batch_size = 256
seed = 0

hypers = {1: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'DDPG'
          2: {'n_critics': 1, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BC'
          3: {'n_critics': 1, 'n_actions': 10, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'CQL'  1.0 10.0
          4: {'n_critics': 2, 'n_actions': 10, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BCQ'
          5: {'n_critics': 2, 'n_actions': 10, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'BEAR'
          7: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'PLAS'
          8: {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}, # 'IQL' 0.7 0.9
          9: {'n_critics': 1, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': 0.9}} # 'ContrastiveRL'

# all_results_sparse = np.ones((6, 3, 8, 3, 100))*-1
# all_results_dense = np.ones((6, 3, 8, 3, 100))*-1# env, expl, algo, seed, time

all_results_sparse = np.ones((6, 3, 1, 3, 100))*-1
all_results_dense = np.ones((6, 3, 1, 3, 100))*-1# env, expl, algo, seed, time

for i, environment in enumerate([0, 1, 2, 3, 4, 5]):
    for j, exploration in enumerate([0, 1, 2]):
        for k, algo_id in enumerate([9]):
            for l, seed in enumerate([0, 1, 2]):

                algo_name = algorithms[algo_id]
                exploration_strategy = exploration_processes[exploration]
                environment_details = experiments[environment]

                if algo_id == 0:
                    tmp_res_s = np.ones((3, 100))*-1
                    tmp_res_d = np.ones((3, 100)) * -1
                    for z_id, z_dim in enumerate([32, 128, 512]):
                        exp_name = environment_details["name"] + "_" + exploration_strategy
                        exp_name += "_" + algo_name + "_BS=" + str(batch_size)
                        exp_name += "_z_dim=" + str(z_dim)
                        exp_name += "_reg=" + str(reg)
                        exp_name += "_seed=" + str(seed)
                        if os.path.exists("./saved_results/" + exp_name + ".npz"):
                            filenpz = np.load("./saved_results/" + exp_name + ".npz")
                            sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                            tmp_res_s[z_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                            tmp_res_d[z_id, :sparse_reward_records.shape[0]] = dense_reward_records
                    all_results_sparse[i, j, k, l] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                    all_results_dense[i, j, k, l] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
                elif algo_id == 3:
                    tmp_res_s = np.ones((3, 100))*-1
                    tmp_res_d = np.ones((3, 100)) * -1
                    for w_id, c_w in enumerate([1.0, 10.0]):
                        exp_name = environment_details["name"] + "_" + exploration_strategy
                        exp_name += "_" + algo_name + "_BS=" + str(batch_size)
                        baseline_hyper = {'n_critics': 1, 'n_actions': 10, 'conservative_weight': c_w, 'expectile': 0.9}
                        exp_name += "_baselines_hyper=" + str(baseline_hyper)
                        exp_name += "_seed=" + str(seed)
                        if os.path.exists("./saved_results/" + exp_name + ".npz"):
                            filenpz = np.load("./saved_results/" + exp_name + ".npz")
                            sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                            tmp_res_s[w_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                            tmp_res_d[w_id, :sparse_reward_records.shape[0]] = dense_reward_records
                    all_results_sparse[i, j, k, l] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                    all_results_dense[i, j, k, l] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
                elif algo_id == 8:
                    tmp_res_s = np.ones((3, 100))*-1
                    tmp_res_d = np.ones((3, 100)) * -1
                    for ex_id, exp in enumerate([0.7, 0.9]):
                        exp_name = environment_details["name"] + "_" + exploration_strategy
                        exp_name += "_" + algo_name + "_BS=" + str(batch_size)
                        baseline_hyper = {'n_critics': 2, 'n_actions': 1, 'conservative_weight': 10.0, 'expectile': exp}
                        exp_name += "_baselines_hyper=" + str(baseline_hyper)
                        exp_name += "_seed=" + str(seed)
                        if os.path.exists("./saved_results/" + exp_name + ".npz"):
                            filenpz = np.load("./saved_results/" + exp_name + ".npz")
                            sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                            tmp_res_s[ex_id, :sparse_reward_records.shape[0]] = sparse_reward_records
                            tmp_res_d[ex_id, :sparse_reward_records.shape[0]] = dense_reward_records
                    all_results_sparse[i, j, k, l] = tmp_res_s[np.argmax(np.max(tmp_res_s, -1))]
                    all_results_dense[i, j, k, l] = tmp_res_d[np.argmax(np.max(tmp_res_s, -1))]
                else:
                    exp_name = environment_details["name"] + "_" + exploration_strategy
                    exp_name += "_" + algo_name + "_BS=" + str(batch_size)
                    baseline_hyper = hypers[algo_id]
                    exp_name += "_baselines_hyper=" + str(baseline_hyper)
                    exp_name += "_seed=" + str(seed)
                    if os.path.exists("./saved_results/" + exp_name + ".npz"):
                        filenpz = np.load("./saved_results/" + exp_name + ".npz")
                        sparse_reward_records, dense_reward_records = filenpz['arr_0'], filenpz['arr_1']
                        all_results_sparse[i, j, k, l, :sparse_reward_records.shape[0]] = sparse_reward_records
                        all_results_dense[i, j, k, l, :sparse_reward_records.shape[0]] = dense_reward_records

                    # print(environment_details["name"], exploration_strategy, algo_name, ": ", np.max(sparse_reward_records))
                    # print(environment_details["name"], exploration_strategy, algo_name, ": ", np.mean(sparse_reward_records[-10:]))

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


for i, env in enumerate([0, 1, 2]):
    for j, expl in enumerate([0, 1, 2]):
        print(experiments[env]["name"] + " " + exploration_processes[expl])
        for k, algo_id in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
            if np.max(all_results[i, j, k, :, :]) < 0:
                print(" - ")
            else:
                max_rewards = np.max(all_results[i, j, k, :, :], -1)
                max_rewards = max_rewards[np.nonzero(np.maximum(0, max_rewards+0.5))]
                mu = np.mean(max_rewards)
                sigma = np.std(max_rewards)
                print("& $"+f'{mu:.2f}'+"$ \scriptsize{$\pm "+f'{sigma:.2f}'+"$}", end=" ")
        print()
    print("-----------------------------------------")

print()

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












