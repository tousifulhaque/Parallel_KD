#!/bin/bash
#SBATCH -A ASC23013 #Submit some more jobs using ASC23014 as well
#SBATCH -J para  # job name
#SBATCH -o para%j.out # name of the output and error file
#SBATCH -N 1    # total number of nodes requested
#SBATCH -n 4    # total number of tasks requested
#SBATCH -p development   # queue name gpu-a100, normal ordevelopment
#SBATCH -t 00:30:00   # expected maximum runtime (hh:mm:ss)

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/Parallel_Multimodal/bin/activate

#Runnning code and calculating execution time
start_time=$(date +%s.%N)
srun python simple_parallel.py
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" |bc)

#writing the elapsed time to csv
echo "$WORLD_SIZE,$elapsed_time" >> execution_time.csv


