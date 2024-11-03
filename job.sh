#!/usr/bin/env bash

#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32

#SBATCH --gres=gpu:1

#SBATCH -C A30  
#SBATCH -t 48:00:00
#SBATCH --mem 32g 

#SBATCH --job-name="Train DQN" 

srun --unbuffered python main.py --train_dqn --num_episodes 200000 --folder d_dqn_5 --lr 0.0001 --batch_size 32 --target_update_freq 100 --epsilon_decay 0.02 --gamma 0.99 