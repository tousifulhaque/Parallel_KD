#!/bin/bash
#SBATCH -A ASC23013 #Submit some more jobs using ASC23014 as well
#SBATCH -J para  # job name
#SBATCH -o para%j.out # name of the output and error file
#SBATCH -N 2    # total number of nodes requested
#SBATCH -n 8    # total number of tasks requested
#SBATCH -p development   # queue name gpu-a100, normal ordevelopment
#SBATCH -t 00:01:00   # expected maximum runtime (hh:mm:ss)

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=8

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source Parallel_Multimodal/bin/activate

srun python hello_worldmpi.py

