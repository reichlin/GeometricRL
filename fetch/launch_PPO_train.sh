
#! /bin/bash

#for r2 in 0.0 0.1 0.5; do
#  for K in 10; do #  2 4 5
#    for init_layer in 0; do
#      for norm in 0 1; do
#        for clip in 0; do
#          for mini_batch in 0; do
#            for c2 in 0.0 0.01 0.1; do #  1 2
#              sbatch --export=r2=$r2,c2=$c2,K=$K,init_layer=$init_layer,norm=$norm,clip=$clip,mini_batch=$mini_batch,learn_var=1 PPO_train.sbatch
#            done
#            sbatch --export=r2=$r2,c2=0.0,K=$K,init_layer=$init_layer,norm=$norm,clip=$clip,mini_batch=$mini_batch,learn_var=0 PPO_train.sbatch
#          done
#        done
#      done
#    done
#  done
#done

for r in  {61..120}; do
  sbatch --export=r=$r PPO_train.sbatch
done
sleep 1000
for r in  {121..160}; do
  sbatch --export=r=$r PPO_train.sbatch
done
sleep 1000
for r in  {160..201}; do
  sbatch --export=r=$r PPO_train.sbatch
done
