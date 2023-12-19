#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|eowyn|khazadum|rivendell|arwen|gondor|galadriel'

#balrog|rivendell|galadriel galadriel|arwen|gondor|belegost

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate offlineRL

export PYTHONPATH=$PYTHONPATH:/Midgard/home/areichlin/GeometricRL

python3 /Midgard/home/areichlin/GeometricRL/fetch/fetch_main.py \
 --exploration ${expl} \
  --environment ${env} \
   --algorithm ${alg} \
    --batch_size ${bs} \
     --policy_type ${policy_type} \
      --z_dim ${z_dim} \
       --K ${K} \
        --var ${var} \
         --R_gamma ${R_gamma} \
          --reg ${reg} \
           --pi_clip ${pi_clip} \
            --n_critics ${n_c} \
             --n_actions ${n_a} \
              --conservative_weight ${c_wei} \
               --expectile ${expectile} \
                --seed ${seed} \
                 --use_images ${use_images}


