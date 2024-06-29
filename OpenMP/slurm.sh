#!/bin/bash
#
#SBATCH --job-name=OpenMP_kmeans
#SBATCH --output=OpenMP_kmeans.log
#SBATCH --err=OpenMP_kmeans_err.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=400

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
make