import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# seed, dim, size
results_dqn = np.zeros((3, 1, 5, 10000))
results_geom = np.zeros((3, 1, 5, 10000))
# results_dqn = np.zeros((3, 4, 1, 3, 10000))
# results_geom = np.zeros((3, 4, 1, 1, 10000))
for c_dim, cube_dim in enumerate([2]):
    for c_size, cube_size in enumerate([10, 20, 30, 40, 50]):
# for c_dim, cube_dim in enumerate([2, 3, 4, 5]):
#     for c_size, cube_size in enumerate([5]):
        batch_size = 256 #128
        data_collection = "complete"
        for seed in [0, 1, 2]:

            tau = 1.0

            for t, tau in enumerate([1.0]): #0.01, 0.1,

                name_exp = data_collection + "_cube_dim=" + str(cube_dim) + "_cube_size=" + str(cube_size) + "_rand_obst="+str(0) + "_bs=" + str(batch_size)
                name_exp = "DQN_" + name_exp + "_tau=" + str(tau)
                name_exp += "_seed=" + str(seed)

                if os.path.exists("./saved_results_complexity/" + name_exp + ".npz"):
                    filenpz = np.load("./saved_results_complexity/" + name_exp + ".npz")
                    results_dqn[seed, c_dim, c_size, :filenpz['arr_0'].shape[0]] = filenpz['arr_0']

            dim_z = 128
            reg = 10.0

            for t, dim_z in enumerate([128]):

                name_exp = data_collection + "_cube_dim=" + str(cube_dim) + "_cube_size=" + str(cube_size) + "_rand_obst="+str(0) + "_bs=" + str(batch_size)
                name_exp = "GeomRL_" + name_exp + "_z_dim=" + str(dim_z) + "_reg=" + str(reg) + "_use_policy=" + str(0)
                name_exp += "_seed=" + str(seed)

                if os.path.exists("./saved_results_complexity/" + name_exp + ".npz"):
                    filenpz = np.load("./saved_results_complexity/" + name_exp + ".npz")
                    results_geom[seed, c_dim, c_size, :filenpz['arr_0'].shape[0]] = filenpz['arr_0']

smoothing_window = 10
# smooth_results_dqn = np.zeros((5, 3, 10000))
# smooth_results_geom = np.zeros((5, 3, 10000))
solutions = np.zeros((2, 1, 5, 3))
# solutions = np.zeros((2, 4, 1, 3, 3))
for i in range(1):
    for j in range(5):
# for i in range(4):
#     for j in range(1):
        #for k in range(3):
        for seed in range(3):
            cum_reward = 0
            solution_time = -1
            for t in range(10000):
                if cum_reward >= 25:
                    results_dqn[seed, i, j, t] = 1
                    if solution_time == -1:
                        solution_time = t
                else:
                    if results_dqn[seed, i, j, t] == 0:
                        cum_reward = 0
                    else:
                        cum_reward += 1
            if solution_time == -1:
                solution_time = 10000
            solutions[0, i, j, seed] = solution_time
for i in range(1):
    for j in range(5):
# for i in range(4):
#     for j in range(1):
        #for k in range(1):
        for seed in range(3):
            cum_reward = 0
            solution_time = -1
            for t in range(10000):
                if cum_reward >= 25:
                    results_geom[seed, i, j, t] = 1
                    if solution_time == -1:
                        solution_time = t
                else:
                    if results_geom[seed, i, j, t] == 0:
                        cum_reward = 0
                    else:
                        cum_reward += 1
            if solution_time == -1:
                solution_time = 10000
            solutions[1, i, j, seed] = solution_time

#             t_min = max(0, t-smoothing_window)
#             t_max = min(1000, t+smoothing_window)
#             avg_t = np.mean(results_dqn[i, j, t_min:t_max])
#             smooth_results_dqn[i, j, t] = avg_t
#
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             for l in range(2):
#                 for t in range(1000):
#                     t_min = max(0, t-smoothing_window)
#                     t_max = min(1000, t+smoothing_window)
#                     avg_t = np.mean(results_geom[i, j, k, l, t_min:t_max])
#                     smooth_results_geom[i, j, k, l, t] = avg_t

print()

plt.plot([10, 20, 30, 40, 50], np.mean(solutions[0, 0, :], -1))
#plt.scatter([10, 20, 30, 40, 50], np.mean(solutions[0, 0, :], -1))
plt.plot([10, 20, 30, 40, 50], np.mean(solutions[1, 0, :], -1))
#plt.scatter([10, 20, 30, 40, 50], np.mean(solutions[1, 0, :], -1))
plt.legend(['DQN', 'GeomRL'])
plt.title("Sample Complexity w.r.t. maze's size")
plt.show()

# TODO: compute best hyper param

print()

plt.plot([2, 3, 4, 5], np.mean(solutions[0, :, 0, 0], -1))
plt.plot([2, 3, 4, 5], np.mean(solutions[1, :, 0, 0], -1))
plt.legend(['DQN', 'GeomRL'])
plt.title("Sample Complexity w.r.t. maze's dimensions")
plt.show()

print()




# plt.plot([2, 3, 4, 5], np.mean(solutions[0, :, 0, 0], -1), label="DQN")
# plt.scatter([2, 3, 4, 5], np.mean(solutions[0, :, 0, 0], -1), marker="D")
# plt.plot([2, 3, 4, 5], np.mean(solutions[1, :, 0, 0], -1), label="GeomRL")
# plt.scatter([2, 3, 4, 5], np.mean(solutions[1, :, 0, 0], -1), marker="D")
# plt.title("Sample Complexity w.r.t. maze's dimensions")
# plt.legend()
# plt.show()
