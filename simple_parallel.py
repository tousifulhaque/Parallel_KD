import torch
import os
import argparse
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler


if __name__ == "__main__":

    # dist.init_process_group("nccl", world_size = 8, rank = 0)
    # rank = dist.get_rank()
    # print(f'Starting rank: {rank}')

    parser = argparse.ArgumentParser()

    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    distributed = args.world_size > 1

    if distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            rank = local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print(f'Rank: {args.rank}')