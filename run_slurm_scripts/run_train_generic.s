#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=60GB
#SBATCH --output=/vast/experiments/dsynth/scripts/seld_run/logs/train_run/final_exp_single_%j.out

local=/tmp/$USER/local
mkdir -p $local

singularity \
    exec --nv\
    --overlay /scratch//envs/dcase/overlay-7.5GB-300K.ext3:ro \
    /scratch/envs/dcase/dcase.sif \
    /bin/bash -c "
source /ext3/env.sh
python3 -u ../revamp_seld/seld-dcase2022_dist/train_seldnet.py $PARAM
"
