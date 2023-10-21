#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=60GB
#SBATCH --output=/vast/sk8974/experiments/dsynth/scripts/seld_run/logs/train_run/final_exp_single_%j.out

local=/tmp/$USER/local
mkdir -p $local

singularity \
    exec --nv\
    --overlay /scratch/sk8974/envs/dcase/overlay-7.5GB-300K.ext3:ro \
    /scratch/sk8974/envs/dcase/dcase.sif \
    /bin/bash -c "
source /ext3/env.sh
python3 -u ../revamp_seld/seld-dcase2022_dist/train_seldnet.py $PARAM
"
