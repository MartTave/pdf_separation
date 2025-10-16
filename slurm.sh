#!/bin/bash
#SBATCH --job-name=marttave-warp-render     # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=05:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                        # number of gpus per node
#SBATCH --output=output/output.txt   # Standard output file
#SBATCH --error=output/error.txt     # Standard error file
# You can change this line to target either ChaCha or Calypso
#SBATCH --partition=Disco

# This is the part that save the stats for the GPU to later analyze
source ./.venv/bin/activate
python -u train_model.py
