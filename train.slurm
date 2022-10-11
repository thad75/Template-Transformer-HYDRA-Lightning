#!/bin/bash
#SBATCH --account=way@v100        # IMPORTANT, préciser le compte à utiliser
#SBATCH --job-name=SAMDETR   # nom du job
#SBATCH --output=SAMDETRwCropDynamicTrain%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=SAMDETRwCropDynamicTrain%j.err  # fichier d'erreur (%j = job ID)
#SBATCH --partition=gpu_p2s    # partition
#SBATCH --qos=qos_gpu-t3         # QoS : Quality of Service
#SBATCH --time=15:00:00           # temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --nodes=1                 # reserver 1 nœud
#SBATCH --ntasks-per-node=8      # Maximum number of tasks on each node
#SBATCH --gres=gpu:8           # reserver 8 GPU
#SBATCH --hint=nomultithread      # desactiver l'hyperthreading
#SBATCH --cpus-per-task=3

module purge                      # nettoyer les modules herites par defaut
module load pytorch-gpu/py3/1.9.0 # charger les modules


set -x                            # activer l'echo des commandes
srun python -u my_app.py # executer son sc