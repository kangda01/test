import os
import numpy as np
import torch


import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset1 import TrainDataset, train_dataset_collate, test_dataset_collate
from train_utils import train_eval_utils as utils



def get_train_lines(train_data):
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(train_data) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines)  - num_val
    # num_train = int(len(lines) * 0.001)
    return lines, num_train, num_val



def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    train_data_set = TrainDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=False)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    drop_last=True,
                                                    collate_fn=train_dataset_collate)

    # load validation data set
    # val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set = TrainDataset(lines[num_train:num_val+num_train], (input_shape[0], input_shape[1]), mosaic=False)
    val_data_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      drop_last=True,
                                                      collate_fn=train_dataset_collate)

    # create model num_classes equal background + 1 classes
    model = create_model(num_classes=2)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    val_loss = []

    learning_rate = []
    val_mAP = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        pro_epoch_total_loss_train = []
        pro_epoch_total_loss_val = []
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, parser_data.epochs, pro_epoch_total_loss_train = pro_epoch_total_loss_train,
                              train_loss=train_loss, train_lr=learning_rate, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.val_one_epoch(model, val_data_loader,
                              device, epoch, parser_data.epochs, pro_epoch_total_loss_val=pro_epoch_total_loss_val,
                              val_loss=val_loss)

        # save weights
        # save_files = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        #     'epoch': epoch}
        # lr = optimizer.param_groups[0]["lr"]
        # torch.save(save_files, "./save_weights/Epoch_%03d_Loss_%.4f_lr_%.6f.pth"% ((epoch+1),
        #                                                                            np.mean(pro_epoch_total_loss_train),
        #                                                                            lr))

        save_files = {
            'model': model,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        lr = optimizer.param_groups[0]["lr"]
        torch.save(save_files, "./save_weights1/Epoch_%03d_Loss_%.4f_lr_%.6f.pth" % ((epoch + 1),
                                                                                    np.mean(pro_epoch_total_loss_train),
                                                                                    lr))


        print('Loss_train: %.4f || Loss_val: %.4f ' % (
            np.mean(pro_epoch_total_loss_train), np.mean(pro_epoch_total_loss_val)))
        print('Saving state, iter:', str(epoch + 1))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from plot_curve import plot_map
        plot_map(val_mAP)



if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    # 因为使用的官方的混合精度训练是1.6.0后才支持的，所以必须大于等于1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    from config import get_parser


    args = get_parser()
    print(args)
    #输入图片的大小
    input_shape = (args.height, args.width)

    #读取txt文件，划分训练验证集
    lines, num_train, num_val = get_train_lines(args.data_path)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
