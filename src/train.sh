#!/bin/bash
#SBATCH --job-name=green
#SBATCH --account=loki
#SBATCH --partition=dc-gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12




module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3 
module load CUDA/12 

source /p/project1/profound/al-zeqri1/DiffDock-dev/DiffDock-PP/diffdock_pp_jureca/bin/activate
export TORCH_HOME=/p/project1/profound/al-zeqri1/.cache/torch


NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=2  # split across all GPUs

NAME="dips_esm"  # change to name of config file
RUN_NAME="large_model_dips" # should uniauely describe the current experiment
CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"

echo SAVE_PATH: $SAVE_PATH

python src/main.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED \
    --project "DiffDock Tuning" \
    --visualize_n_val_graphs 0 \
    --visualization_path $VISUALIZATION_PATH \
    --translation False\
    --rotation False\
    --latent True\
    --tr_weight 0\
    --rot_weight 0\
    --latent_weight 1\
    #--checkpoint_path $SAVE_PATH \
    #--debug True # load small dataset
    #--entity coarse-graining-mit \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path