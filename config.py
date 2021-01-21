import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # parser.add_argument('--device', default='cpu', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data_path', default='image_labels_split9920.txt', help='dataset')
    parser.add_argument('--input_dir', default=r'E:\kaggle\nfl-impact-detection\clip_images_992', help='input_dir')

    # 输入模型的shape,这两个值不需要，因为faster rcnn最终会使用roipooling，不需要固定图的大小。
    parser.add_argument('--height', default=992, type=int, help='input height')
    parser.add_argument('--width', default=992, type=int, help='input width')
    # 权重保存地址
    parser.add_argument('--output_dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='./save_weights/Epoch_014_Loss_0.1148_lr_0.000545.pth', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    # 优化器
    parser.add_argument('--TRAIN_OPTIMIZER', default='adam', type=str, help='optimizer')

    # 初始学习率
    parser.add_argument('--lr', default=0.01, type=float, help='lr')

    # 正则化系数
    parser.add_argument('--weight_decay', default=0.00, type=float, help='l2 regularization')

    # momentum
    parser.add_argument('--momentum', default=0.90, type=float, help='momentum')

    #评估的模型加载地址
    parser.add_argument('--pth_path', default='./save_weights/Epoch_012_Loss_0.1127_lr_0.000545.pth', type=str, help='pth_path')

    #计算mAP的一些参数
    parser.add_argument('--save_err_miss', default=False, type=bool, help='save_err_miss')
    # parser.add_argument('--input_dir', default='', type=str, help='input_dir')




    return parser.parse_args()