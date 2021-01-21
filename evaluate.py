import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset1 import TestDataset,  test_dataset_collate
import math
from Evaluation.map_eval_pil import compute_map
from easydict import EasyDict



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

def load_model_pth(model, pth, cut=None):
    print('Loading weights into state dict, name: %s'%(pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    checkpoint = torch.load(pth, map_location=device)
    pretrained_dict = checkpoint['model']

    match_dict = {}
    print_dict = {}
    if cut== None: cut = (len(pretrained_dict) - 1)
    try:
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if i <= cut:
                assert np.shape(model_dict[k]) == np.shape(v)
                match_dict[k] = v
                print_dict[k] = v
            else:
                print_dict[k] = '[NO USE]'

    except:
        print('different shape with:', np.shape(model_dict[k]), np.shape(v), '  name:', k)
        assert 0

    for i, key in enumerate(print_dict):
        value = print_dict[key]
        print('items:', i, key, np.shape(value) if type(value) != str else value)

    model_dict.update(match_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    # width = img.shape[1]
    # height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[int(cls_id)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def make_labels_and_compute_map(infos, input_dir, save_err_miss=False):
    out_lines,gt_lines = [],[]
    out_path = 'Evaluation/out.txt'
    gt_path = 'Evaluation/true.txt'
    foutw = open(out_path, 'w')
    fgtw = open(gt_path, 'w')
    for info in infos:
        for t, j, k in zip(info[0], info[1], info[2]):
            pre_boxes, pre_labels, pre_scores = t['boxes'], t['labels'], t['scores']
            shapes = k
            for i, images in enumerate([pre_boxes.tolist()]):
                for box in images:
                    # bbx = [box[0]*shapes[i][1], box[1]*shapes[i][0], box[2]*shapes[i][1], box[3]*shapes[i][0]]
                    bbx = str(box)
                    cls = str(pre_labels[i])
                    prob = str(pre_scores[i])
                    img_name = os.path.split([shapes][i][2])[-1]
                    line = '\t'.join([img_name, 'Out:', cls, prob, bbx])+'\n'
                    out_lines.append(line)

            gt_boxes, gt_labels = j['boxes'], j['labels']
            for i, images in enumerate([gt_boxes.tolist()]):
                for box in images:
                    bbx = str(box)
                    cls = str(gt_labels[i])
                    img_name = os.path.split([shapes][i][2])[-1]
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



@torch.no_grad()
def main(parser_data, lines, draw=True ):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device evaluating.".format(device.type))

    # 加载数据
    evaluate_data_set = TestDataset(lines[num_train:num_train+num_val], (parser_data.height, parser_data.width))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    evaluate_data_loader = torch.utils.data.DataLoader(evaluate_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=nw,
                                                    drop_last=True,
                                                    collate_fn=test_dataset_collate)
    # 加载模型
    model = create_model(num_classes=2)
    model = load_model_pth(model, parser_data.pth_path)

    model.to(device)
    model.eval()

    # 显示进度条
    evaluate_data_loader = tqdm(evaluate_data_loader)
    infos = []
    for i, batch in enumerate(evaluate_data_loader):
        images_src, images, targets, shapes = batch[0], batch[1], batch[2], batch[3]
        images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        # model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device).numpy() for k, v in t.items()} for t in outputs]

        infos.append([outputs, targets, shapes])
        if draw:
            for x in range(len(outputs)):
                os.makedirs('result1', exist_ok=True)
                savename = os.path.join('result1', os.path.split(shapes[x][2])[-1])
                plot_boxes_cv2(images_src[x], outputs[x]['boxes'], savename=savename, class_names=None)
        torch.set_num_threads(n_threads)

    Map = make_labels_and_compute_map(infos, parser_data.input_dir, save_err_miss=parser_data.save_err_miss)
    return Map



if __name__ == "__main__":
    from config import get_parser
    args = get_parser()
    lines, num_train, num_val = get_train_lines(args.data_path)
    args.pth_path = './save_weights/Epoch_022_Loss_0.0736_lr_0.000059.pth'
    main(args, lines, draw=True)


