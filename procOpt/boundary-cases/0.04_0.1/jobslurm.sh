#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=jet_rans
#SBATCH --output=slurm.out
module load ansys/fluent-21.2.0
cd $SLURM_SUBMIT_DIR
time fluent 2ddp -g -pdefault -t$SLURM_NTASKS -slurm -i jet_rans-axi_sym.jou > run.out