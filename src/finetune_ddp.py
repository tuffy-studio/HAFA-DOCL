import os
import csv
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from stats_calculation import calculate_stats
import torch.distributed as dist


torch.autograd.set_detect_anomaly(False) # 若为True，则开启异常检测，追踪模型发散原因，但会影响训练速度

def save_data(csv_file, epoch, data, data_name):
    # 如果 CSV 文件不存在，就创建并写入表头
    try:
        # 打开 CSV 文件，追加模式（'a'）避免覆盖原数据
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 如果文件为空，写入表头
            if file.tell() == 0:  # 检查文件是否为空
                writer.writerow(['epoch', data_name])  # 表头

            # 写入当前 epoch 和损失值
            writer.writerow([epoch, data])
    except Exception as e:
        print(f"Error while saving data: {e}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, train_sampler, test_loader, test_sampler, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start training model on {device}")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    best_epoch, best_auc = 0, -np.inf
    global_step, epoch = 0, 0
    save_dir = args.save_dir
    
    model.to(device)

    mlp_params = []
    base_params = []

    for name, param in model.named_parameters():
        if 'visual_encoder' in name:
            base_params.append(param)
        else:
            mlp_params.append(param)
    
    trainables = [p for p in model.parameters() if p.requires_grad]

    if dist.get_rank() == 0:
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

        print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
        print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    

    optimizer = torch.optim.AdamW([{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-2}, 
                                  {'params': mlp_params, 'lr': args.lr * args.head_lr_ratio, 'weight_decay':5e-2}], 
                                   betas=(0.95, 0.999))
    
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    if dist.get_rank() == 0:
        print('base lr, mlp lr : ', base_lr, mlp_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,       # 第一个周期 10 epoch
    T_mult=1,     # 倍增周期长度
    eta_min=1e-9,  # 最小学习率（防止为 0）
    last_epoch=-1
    )  

    class_weights = torch.tensor([1.0, 1.0]).to(device)
    CE_loss_fn = nn.CrossEntropyLoss(weight=class_weights)#, label_smoothing=0.01)

    scaler = GradScaler()
    
    if dist.get_rank() == 0:
        print("current #steps=%s, #epochs=%s" % (global_step, epoch))
        print("start training...")

    model.train()

    # 创建日志目录，根据当前时间戳命名
    start_time = datetime.datetime.now() + datetime.timedelta(hours=0)
    start_time_str = start_time.strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join("./logs/finetuning/", start_time_str)
    os.makedirs(log_dir, exist_ok=True)

    optimizer.zero_grad() 

    use_amp = args.use_amp   # True = 使用混合精度 (autocast + GradScaler)，False = 全精度 FP32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # 如果 use_amp=False，scaler 不会起作用
    verbose = args.verbose

    epoch = 1
    while epoch < args.n_epochs + 1:
        model.train()
        if dist.get_rank() == 0:
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        train_sampler.set_epoch(epoch)
        for i, (imgs, origin_imgs, labels) in enumerate(train_loader):
            B = imgs.shape[0]
            imgs = imgs.to(device)
            origin_imgs = origin_imgs.to(device)
            if verbose == True and i % 10 == 0 and dist.get_rank() == 0:
                print(f"epoch: {epoch}, train number:{i}", flush=True)
                #print(f"labels: {labels}")
            labels = labels.to(device)
            
            # 使用 autocast 来进行混合精度前向计算
            with autocast(enabled=use_amp):
                if args.moe:
                    outputs, visual_agg = model(imgs)
                    origin_outputs, origin_visual_agg = model(origin_imgs)
                else:
                    outputs, visual_agg_3, visual_agg_6, visual_agg_9, visual_agg_12 = model(imgs)
                    origin_outputs, origin_visual_agg_3, origin_visual_agg_6, origin_visual_agg_9, origin_visual_agg_12 = model(origin_imgs)
                    _ = origin_visual_agg_3.mean() + origin_visual_agg_6.mean() + origin_visual_agg_9.mean() + origin_visual_agg_12.mean()
                
                # CE loss
                degraded_loss = CE_loss_fn(outputs, labels)
                origin_loss = CE_loss_fn(origin_outputs, labels)

                # 让原始图像的特征聚合结果与降质图像的特征聚合结果更接近
                if args.moe:
                    cos_sim = F.cosine_similarity(origin_visual_agg.detach(), visual_agg, dim=1)
                    consistency_loss = 1 - cos_sim.mean()
                else:
                    cos_sim_3 = F.cosine_similarity(origin_visual_agg_3.detach(), visual_agg_3, dim=1)
                    cos_sim_6 = F.cosine_similarity(origin_visual_agg_6.detach(), visual_agg_6, dim=1)
                    cos_sim_9 = F.cosine_similarity(origin_visual_agg_9.detach(), visual_agg_9, dim=1)
                    cos_sim_12 = F.cosine_similarity(origin_visual_agg_12.detach(), visual_agg_12, dim=1)
                    consistency_loss = 4 - cos_sim_3.mean() - cos_sim_6.mean() - cos_sim_9.mean() - cos_sim_12.mean()

                loss = 2 * degraded_loss + origin_loss + 0.05 * consistency_loss

                if verbose == True and i % 10 == 0 and dist.get_rank() == 0:
                    print(f"degraded_loss: {degraded_loss}, origin_loss: {origin_loss}, consistency_loss: {consistency_loss}", flush=True)
                    print(f"loss: {loss}", flush=True)
                    #print(f"outputs: {torch.softmax(outputs, dim=-1, dtype=torch.float16).cpu().detach()}")

            # 使用 GradScaler 缩放损失并进行反向传播
            scaler.scale(loss).backward()

            # === 添加梯度裁剪 ===
            scaler.unscale_(optimizer)  # unscale 是关键
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            # ====================

            # 使用 GradScaler 更新参数
            scaler.step(optimizer)

            # 更新 GradScaler 的状态
            scaler.update()

            optimizer.zero_grad()

            # loss_av is the main loss
            loss_meter.update(loss.item(), B)

            if np.isnan(loss_meter.avg):
                print("training diverged...")
                return

            global_step += 1

            if args.save_model == True and i % 100 == 0 and i > 99 and dist.get_rank() == 0:
                torch.save(model.module.state_dict(), "%s/models/model.%d.pth" % (save_dir, epoch))
                torch.save(optimizer.state_dict(), "%s/models/optimizer.%d.pth" % (save_dir, epoch))

        # 学习率调度器更新
        scheduler.step()

        if args.save_model == True and dist.get_rank() == 0:
            torch.save(model.module.state_dict(), "%s/models/model.%d.pth" % (save_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optimizer.%d.pth" % (save_dir, epoch))
        
        if dist.get_rank() == 0:
            save_data(f"{log_dir}/train_loss_ft.csv", epoch=epoch, data=loss_meter.avg, data_name="train_loss")

        
        #========================================模型验证=================================
        if dist.get_rank() == 0:
            print('start validation')
        
        stats, valid_loss = validate(model, test_loader, test_sampler, args)

        ap = stats[1]['ap']
        auc = stats[1]['auc']
        acc = stats[1]['acc']

        # 打印验证结果
        if dist.get_rank() == 0:
            print("============================================")
            print(f"Finetuning epoch: {epoch} ")
            print("validation finished")
            print("ACC: {:.6f}".format(acc))
            print("AUC: {:.6f}".format(auc))
            print("AP: {:.6f}".format(ap))
            print("============================================")

            # 保存验证结果到 CSV 文件
            save_data(f"{log_dir}/test_loss_ft.csv", epoch=epoch, data=valid_loss, data_name="test_loss")
            save_data(f"{log_dir}/AP.csv", epoch=epoch, data=ap, data_name="ap")
            save_data(f"{log_dir}/ACC.csv", epoch=epoch, data=acc, data_name="acc")
            save_data(f"{log_dir}/AUC.csv", epoch=epoch, data=auc, data_name="auc")
        
        if auc > best_auc:
            best_epoch = epoch
            best_acc = acc
            best_auc = auc
        elif auc == best_auc:
            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc

        # 保存模型参数
        if best_epoch == epoch and dist.get_rank() == 0:
            torch.save(model.module.state_dict(), "%s/models/best_model.pth" % (save_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (save_dir))
        
        print('Epoch {0} learning rate: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        epoch += 1

        # 每个epoch重置计数类
        loss_meter.reset()



def validate(model, val_loader, val_sampler, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = torch.tensor([1.0, 1.0]).to(device)
    CE_loss_fn = nn.CrossEntropyLoss(weight=class_weights)#, label_smoothing=0.01)

    model = model.to(device)
    model.eval()

    A_predictions, A_targets, A_loss = [], [], []

    with torch.no_grad():
        for i, (imgs, origin_imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            origin_imgs = origin_imgs.to(device)
            if args.verbose == True and dist.get_rank() == 0:
                print(f"validate number: {i}")
                #print(f"labels: {labels}")
            A_targets.append(labels)
            labels = labels.to(device)
 
            with autocast(enabled=False):

                if args.moe:
                    outputs, visual_agg = model(imgs)
                    origin_outputs, origin_visual_agg = model(origin_imgs)
                else:
                    outputs, visual_agg_3, visual_agg_6, visual_agg_9, visual_agg_12 = model(imgs)
                    origin_outputs, origin_visual_agg_3, origin_visual_agg_6, origin_visual_agg_9, origin_visual_agg_12 = model(origin_imgs)
                    _ = origin_visual_agg_3.mean() + origin_visual_agg_6.mean() + origin_visual_agg_9.mean() + origin_visual_agg_12.mean()

                loss = CE_loss_fn(outputs, labels) # CE loss
                origin_loss = CE_loss_fn(origin_outputs, labels)

                if args.moe:
                    cos_sim = F.cosine_similarity(origin_visual_agg.detach(), visual_agg, dim=1)
                    consistency_loss = 1 - cos_sim.mean()
                else:
                    cos_sim_3 = F.cosine_similarity(origin_visual_agg_3.detach(), visual_agg_3, dim=1)
                    cos_sim_6 = F.cosine_similarity(origin_visual_agg_6.detach(), visual_agg_6, dim=1)
                    cos_sim_9 = F.cosine_similarity(origin_visual_agg_9.detach(), visual_agg_9, dim=1)
                    cos_sim_12 = F.cosine_similarity(origin_visual_agg_12.detach(), visual_agg_12, dim=1)
                    consistency_loss = 4 - cos_sim_3.mean() - cos_sim_6.mean() - cos_sim_9.mean() - cos_sim_12.mean()

            predictions = outputs.to('cpu').detach()

            if args.verbose == True and dist.get_rank() == 0:
                print(f"loss: {loss}")
                print(f"origin_loss: {origin_loss}")
                print(f"consistency_loss: {consistency_loss}")
                #print(f"outputs: {predictions}")

            A_predictions.append(predictions)

            A_loss.append(loss.to('cpu').detach())


        output = torch.cat(A_predictions)
        target = torch.cat(A_targets)

        loss = np.mean(A_loss)

        target = F.one_hot(target, num_classes=2).float() #muliti-class
        stats = calculate_stats(torch.softmax(output, dim=-1).cpu(), target.cpu())

        return stats, loss