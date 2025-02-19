#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=i64m1tga800u
#SBATCH -J loretta
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=low

source /hpc2ssd/softwares/anaconda3/bin/activate loretta
/hpc2hdd/home/yzuo099/loretta-1/bert_model/run_all_bert_exp.sh 

# sbatch --partition=i64m1tga800u --gres=gpu:1 -J lorreta -o output_test.txt -e error.txt task_hpc.sh
# squeue -u yzuo099