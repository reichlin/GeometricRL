
#! /bin/bash

# DATA COLLECTION
#for env in 0 1 2; do
#  for expl in 0 1; do
#    sbatch --export=name_id=0,expl=$expl,env=$env OfflineRL.sbatch
#  done
#done
#for env in 3 4 5; do
#  for expl in 0 1; do
#    for name_id in 0 1 2 3 4 5 6 7 8 9; do
#      sbatch --export=name_id=$name_id,expl=$expl,env=$env OfflineRL.sbatch
#    done
#  done
#done

# D4RL
# 10 baselines, 24 GeomRL
for seed in 0 1 2; do #  1 2
  for expl in 0 1 2; do
    for env in 0 1 2; do #  2 4 5
#      # GeometricRL
#      for z_dim in 32 128 512; do
#        for reg in 1.0; do
#          for batch_size in 256; do
#            sbatch --export=expl=$expl,env=$env,alg=0,bs=$batch_size,policy_type=1,z_dim=$z_dim,K=1,var=1.0,R_gamma=1.0,reg=$reg,pi_clip=-1,n_c=1,n_a=1,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#          done
#        done
#      done
#      # DDPG
#      for n_c in 2; do
#        sbatch --export=expl=$expl,env=$env,bs=256,alg=1,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=1,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#      done
#      # BC
#      sbatch --export=expl=$expl,env=$env,bs=256,alg=2,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=1,n_a=1,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#      # CQL
#      for n_a in 10; do
#        for c_wei in 1.0 10.0; do
#          sbatch --export=expl=$expl,env=$env,bs=256,alg=3,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=1,n_a=$n_a,c_wei=$c_wei,expectile=0.9,seed=$seed OfflineRL.sbatch
#        done
#      done
#      # BCQ
#      for n_c in 2; do
#        for n_a in 10; do
#          sbatch --export=expl=$expl,env=$env,bs=256,alg=4,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=$n_a,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#        done
#      done
#      # BEAR
#      for n_c in 2; do
#        for n_a in 10; do
#          sbatch --export=expl=$expl,env=$env,bs=256,alg=5,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=$n_a,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#        done
#      done
##      # AWAC
##      for n_c in 2; do
##        for n_a in 10; do
##          sbatch --export=expl=$expl,env=$env,alg=6,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=$n_a,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
##        done
##      done
#      # PLAS
#      for n_c in 2; do
#        sbatch --export=expl=$expl,env=$env,bs=256,alg=7,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=1,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
#      done
#      # IQL
#      for n_c in 2; do
#        for expectile in 0.7 0.9; do
#          sbatch --export=expl=$expl,env=$env,bs=256,alg=8,policy_type=1,z_dim=128,K=1,var=0.1,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=$n_c,n_a=1,c_wei=10.0,expectile=$expectile,seed=$seed OfflineRL.sbatch
#        done
#      done
#      sleep 3600
      # ContrastiveRL
      sbatch --export=expl=$expl,env=$env,bs=256,alg=9,policy_type=1,z_dim=128,K=1,var=1.0,R_gamma=1.0,reg=1.0,pi_clip=-1.0,n_c=1,n_a=1,c_wei=10.0,expectile=0.9,seed=$seed OfflineRL.sbatch
    done
  done
done



# ContrastiveRL
# QuasiMetric
#sbatch --export=exploration=$exploration,environment=$environment,algorithm=10,batch_size=$batch_size,z_dim=32,reg=1,R_gamma=1,n_critics=1,n_actions=10,conservative_weight=5.0,expectile=0.7,seed=$seed OfflineRL.sbatch






