#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=20:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=dcase_mp_1
#SBATCH --output=../logs/dcase_synth_%j.out

local=/tmp/$USER/local
mkdir -p $local

singularity \
    exec \
    --overlay /scratch/sk8974/envs/dsynth/dsynth.ext3:ro \
    /scratch/sk8974/envs/dsynth/dsynth.sif \
    /bin/bash -c "
source /ext3/env.sh
python -u example_script_DCASE2022.py"
