import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np 
import random
import os
import argparse
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
# from torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms

def set_random_seeds(random_seed = 0):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, 
                        help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--random-seed', type =int, default = 0, help ='Seed for weight genreation' )

    parser.add_argument('--lr', default = 0.1 , type = float, help = 'Learning rate for model')

    parser.add_argument('--epoch', default = 100, type = int, help = 'Epoch number')

    parser.add_argument('--batch-size', default = 16, type = int)
    args = parser.parse_args()
    return args


def evaluate(model , test_loader):
    model.eval()

    correct = 0
    total = 0 
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0], data[1]
            output = model(images)
            _, pred = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (pred ==labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    args = arg_parse()
    set_random_seeds(random_seed = args.random_seed)
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

    model = torchvision.models.resnet18(weights = None)

    #Encapsulate the model with DDP
    ddp_model = DDP(model, device_ids = None, output_device = None)

    #Intiate the dataset and distributed sampler
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train = True, download = False, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root = "data", train = False, download = False, transform = transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset= train_set)

    train_loader = DataLoader(dataset = train_set, batch_size = args.batch_size, sampler =train_sampler, num_workers = 8)
    test_loader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = False, num_workers = 8)

    #loss function 
    criterion = nn.CrossEntropyLoss()

    #optimizer 
    optimizer = optim.SGD(ddp_model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-5)

    #training loop 
    for epoch in range(args.epoch):

        #Save and evaluate model 
        if epoch % 10 == 0 : 
            if args.rank == 0 :
                accuracy = evaluate(model = ddp_model, test_loader = test_loader)
                print("-"* 75)
                print(f'Local rank {args.rank}, Epoch: {epoch+1}, Val_Acc: {accuracy}')
        
        ddp_model.train()
        total = 0
        correct = 0 
        train_acc = 0
        for data in train_loader:
            inputs, labels = data[0], data[1]
            total += labels.size(0) 
            optimizer.zero_grad()
            output = ddp_model(inputs)
            _, pred = torch.max(output.data, 1)
            correct += (pred == labels).sum().item()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_acc  = round( correct / total , 3) 
            print(f'Local rank {args.rank}, Epoch{epoch}, Train Acc {train_acc}')

if __name__ == "__main__":

    main()




    

 