
#! /bin/bash


## Computational_complexity depth
#for seed in 0 1 2; do
#  for cube_dim in 2; do
#    for cube_size in 10 20 30 40 50; do
#      # DQN
#      for tau in 1.0; do
#        sbatch --export=model_type=1,cube_dim=$cube_dim,cube_size=$cube_size,data_collection=2,batch_size=256,tau=$tau,z_dim=32,reg=0.1,seed=$seed CCRL.sbatch
#      done
#      # GeomRL
#      for z_dim in 128; do
#        for reg in 10.0; do
#          sbatch --export=model_type=0,cube_dim=$cube_dim,cube_size=$cube_size,data_collection=2,batch_size=256,tau=0.1,z_dim=$z_dim,reg=$reg,seed=$seed CCRL.sbatch
#        done
#      done
#    done
#  done
#  #sleep 3600
#done
#sleep 3600
# Computational_complexity dimensions
for seed in 0 1 2; do
  for cube_dim in 2 3 4 5; do
    for cube_size in 5; do
#      # DQN
      for tau in 1.0 0.1 0.01; do
        sbatch --export=model_type=1,cube_dim=$cube_dim,cube_size=$cube_size,data_collection=2,batch_size=128,tau=$tau,z_dim=32,reg=0.1,seed=$seed CCRL.sbatch
      done
      # GeomRL
      for z_dim in 128; do
        for reg in 10.0; do
          sbatch --export=model_type=0,cube_dim=$cube_dim,cube_size=$cube_size,data_collection=2,batch_size=128,tau=0.1,z_dim=$z_dim,reg=$reg,seed=$seed CCRL.sbatch
        done
      done
    done
  done
done



