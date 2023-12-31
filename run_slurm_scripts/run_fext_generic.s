#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --output=feature_ext_%j.out

local=/tmp/$USER/local
mkdir -p $local

singularity \
    exec \
    --overlay /scratch/envs/dcase/overlay-7.5GB-300K.ext3:ro \
    /scratch/sk8974/envs/dcase/dcase.sif \
    /bin/bash -c "
source /ext3/env.sh
python3 -u ../revamp_seld/seld-dcase2022_dist/batch_feature_extraction.py $PARAM
"