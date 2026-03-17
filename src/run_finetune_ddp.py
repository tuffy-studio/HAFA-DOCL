import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader import FineTuneDataset
from models.HiViT import HiViT_FT
from finetune_ddp import train
import numpy as np
# DDP 环境配置
import torch
import torch.distributed as dist
import os

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# 解析命令行参数
parser = argparse.ArgumentParser(description='Finetune Stage')

parser.add_argument('--data_train', default="/home/home/jielun/HAVIC/CVPR2026workshop/video_data_engine/csvs/finetune_img.csv", type=str, help='path to train data csv')
parser.add_argument('--data_val', default="/home/home/jielun/HAVIC/CVPR2026workshop/video_data_engine/csvs/finetune_img.csv", type=str, help='path to val data csv')

parser.add_argument('--restart', action='store_true', help='Whether to restart training.')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--head_lr_ratio', default=10, type=int, help='learning rate ratio for the head')
parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--use_amp', action='store_true', help='Whether to use mixed precision training.')
parser.add_argument('--verbose', action='store_true', help='Whether to print verbose training logs.')
parser.add_argument('--use_hierarchical', action='store_true', help='Whether to use hierarchical aggregation in the model.')
parser.add_argument('--moe', action='store_true', help='Whether to use Mixture of Experts in the model.')
parser.add_argument('--encoder_embed_dim', default=768, type=int, help='the embedding dimension of the visual encoder, should be 768 for vit_base and 1024 for vit_large')

parser.add_argument('--pretrain_path', default="/home/home/jielun/HAVIC/CVPR2026workshop/weights/fsfm-base.pth", type=str, help='path to pretrain model')
parser.add_argument('--save_model', action='store_true', help='Whether to save model checkpoints.')
parser.add_argument('--save_dir', default='checkpoints', type=str, help='directory to save checkpoints')


args = parser.parse_args()

# mean=[0.532625138759613, 0.4048449993133545, 0.3708747327327728]
# std=[0.25850796699523926, 0.21054500341415405, 0.20785294473171234]

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# 构造数据集
train_dataset = FineTuneDataset(args.data_train, data_augment=True, mean=mean, std=std, if_normalize=True)
val_dataset = FineTuneDataset(args.data_val, data_augment=True, mean=mean, std=std, if_normalize=True)

args.ratio = train_dataset.get_real_fake_ratio()

if dist.get_rank() == 0:
    print(f"Using Train: {len(train_dataset)}, Eval: {len(val_dataset)}")
    print(f"real/fake samples ratio: {args.ratio}")

train_sampler = DistributedSampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=192,   # 512 / 4
    sampler=train_sampler,
    num_workers=16,
    pin_memory=False
)

val_sampler = DistributedSampler(val_dataset)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,   # 512 / 4
    sampler=val_sampler,
    num_workers=16,
    pin_memory=False
)


# 构造模型并加载预训练权重
ft_model = HiViT_FT(use_hierarchical=args.use_hierarchical, moe=args.moe, encoder_embed_dim=args.encoder_embed_dim)

# init model
if args.pretrain_path is not None:
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    if args.restart:
        miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
    else:
        miss, unexpected = ft_model.visual_encoder.load_state_dict(mdl_weight["model"], strict=False)
    if dist.get_rank() == 0:
        print("Missing: ", miss)
        print("Unexpected: ", unexpected)
        print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
else:
    if dist.get_rank() == 0:
        print("Note you are finetuning a model without any pretraining.")


device = torch.device(f"cuda:{local_rank}")
ft_model = ft_model.to(device)

# 模型设置DDP
ft_model = torch.nn.parallel.DistributedDataParallel(ft_model, device_ids=[local_rank], find_unused_parameters=True)

# 开始训练
if dist.get_rank() == 0:
    print("Now start training for %d epochs"%args.n_epochs)
train(ft_model, train_loader, train_sampler, val_loader, val_sampler, args)