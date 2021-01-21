import math
import sys
import os
import time
from tqdm import tqdm
import torch
import numpy as np
# from torch.autograd import Variable
from train_utils.coco_utils import get_coco_api_from_dataset
from train_utils.coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils
from Evaluation.map_eval_pil import compute_map
from easydict import EasyDict


#调用Evaluation模块, 进行map计算和类别准召率计算
def make_labels_and_compute_map(infos, classes, input_dir, save_err_miss=False):
    out_lines,gt_lines = [],[]
    out_path = 'Evaluation/out.txt'
    gt_path = 'Evaluation/true.txt'
    foutw = open(out_path, 'w')
    fgtw = open(gt_path, 'w')
    for info in infos:
        out, gt, shapes = info
        for i, images in enumerate(out):
            for box in images:
                bbx = [box[0]*shapes[i][1], box[1]*shapes[i][0], box[2]*shapes[i][1], box[3]*shapes[i][0]]
                bbx = str(bbx)
                cls = classes[int(box[6])]
                prob = str(box[4])
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, prob, bbx])+'\n'
                out_lines.append(line)

        for i, images in enumerate(gt):
            for box in images:
                bbx = str(box.tolist()[0:4])
                cls = classes[int(box[4])]
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, '1.0', bbx])+'\n'
                gt_lines.append(line)

    foutw.writelines(out_lines)
    fgtw.writelines(gt_lines)
    foutw.close()
    fgtw.close()

    args = EasyDict()
    args.annotation_file = 'Evaluation/true.txt'
    args.detection_file = 'Evaluation/out.txt'
    args.detect_subclass = False
    args.confidence = 0.2
    args.iou = 0.3
    args.record_mistake = True
    args.draw_full_img = save_err_miss
    args.draw_cut_box = False
    args.input_dir = input_dir
    args.out_dir = 'out_dir'
    Map = compute_map(args)
    return Map



def train_one_epoch(model, optimizer, data_loader, device, epoch, Epoch, pro_epoch_total_loss_train=None,
                    train_loss=None, train_lr=None, warmup=False):
    model.train()

    #显示进度条
    data_loader = tqdm(data_loader, desc=f'Train_Epoch {epoch + 1}/{Epoch}')
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    enable_amp = True if "cuda" in device.type else False
    # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    for iteration, batch in enumerate(data_loader):
        images, targets = batch[0], batch[1]

        # with torch.no_grad():
        #     images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).to(device)
        #     targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).to(device)
            # targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).to(device) for ann in targets]

        images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        # images = list(torch.from_numpy(image).to(device) for image in images)
        # targets = list(torch.from_numpy(target).type(torch.FloatTensor).to(device) for target in targets)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=enable_amp):
            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values() if math.isfinite(loss))
            losses_reduced = losses_reduced.float()
            loss_value = losses_reduced.item()

            if isinstance(train_loss, list):
                # 记录训练损失
                train_loss.append(loss_value)

            pro_epoch_total_loss_train.append(loss_value)


            if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # if not math.isfinite(loss_value):
            #     # optimizer.zero_grad()        # 当计算的损失为无穷大时跳过本次训练
            #     continue


        # optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=now_lr)
        if isinstance(train_lr, list):
            train_lr.append(now_lr)


        data_loader.set_postfix(**{'losses_reduced': losses_reduced,
                                 'loss_total': np.mean(pro_epoch_total_loss_train),
                                 'lr': now_lr})
        data_loader.update(1)


@torch.no_grad()
def val_one_epoch(model,  data_loader, device, epoch, Epoch, pro_epoch_total_loss_val=None,
                    val_loss=None):
    # model.eval()

    #显示进度条
    data_loader = tqdm(data_loader, desc=f'Val_Epoch {epoch + 1}/{Epoch}')

    enable_amp = True if "cuda" in device.type else False
    # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    for iteration, batch in enumerate(data_loader):
        images, targets = batch[0], batch[1]

        # with torch.no_grad():
        #     images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).to(device)
        #     targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).to(device)
            # targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).to(device) for ann in targets]

        images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        # images = list(torch.from_numpy(image).to(device) for image in images)
        # targets = list(torch.from_numpy(target).type(torch.FloatTensor).to(device) for target in targets)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss_dict = model(images, targets)
            # losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values() if math.isfinite(loss))
            losses_reduced = losses_reduced.float()
            loss_value = losses_reduced.item()

            if isinstance(val_loss, list):
                # 记录训练损失
                val_loss.append(loss_value)

            pro_epoch_total_loss_val.append(loss_value)


            if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)


        data_loader.set_postfix(**{'losses_reduced': losses_reduced,
                                 'loss_total': np.mean(pro_epoch_total_loss_val)})
        data_loader.update(1)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
