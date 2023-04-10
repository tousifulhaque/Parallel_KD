#!/bin/bash
#SBATCH -A Scalable-Supercomput
#SBATCH -J producer      # job name
#SBATCH -o producer.%j   # name of the output and error file
#SBATCH -N 2               # total number of nodes requested
#SBATCH -n 16                # total number of tasks requested
#SBATCH -p development            # queue name normal or development
#SBATCH -t 00:02:30         # expected maximum runtime (hh:mm:ss)

mpirun -np 4 \
-H 104.171.200.62:2,104.171.200.182:2 \
-x MASTER_ADDR=104.171.200.62 \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 parallel.py --backend=nccl --use_syn --batch_size=8192 --arch=resnet152
