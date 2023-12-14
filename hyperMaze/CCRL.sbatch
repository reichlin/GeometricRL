#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|rivendell|shelob|arwen|gondor|smaug|khazadum|shire|belegost'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate offlineRL

export PYTHONPATH=$PYTHONPATH:/Midgard/home/areichlin/GeometricRL

python3 /Midgard/home/areichlin/GeometricRL/Computational_complexity/main_comp_complex.py \
 --model_type ${model_type} \
  --cube_dim ${cube_dim} \
   --cube_size ${cube_size} \
    --data_collection ${data_collection} \
     --batch_size ${batch_size} \
      --tau ${tau} --z_dim ${z_dim} \
       --reg ${reg} \
        --seed ${seed}




