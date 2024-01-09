#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|eowyn|khazadum|rivendell|arwen|gondor|galadriel'

#balrog|rivendell|galadriel galadriel|arwen|gondor|belegost

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate offlineRL

export PYTHONPATH=$PYTHONPATH:/Midgard/home/areichlin/GeometricRL

python3 /Midgard/home/areichlin/GeometricRL/fetch/data_collection.py \
 --reward ${r}
#  --c2 ${c2} \
#   --K ${K} \
#    --init_layer ${init_layer} \
#     --norm ${norm} \
#      --clip ${clip} \
#        --mini_batch ${mini_batch} \
#        --learn_var ${learn_var}

