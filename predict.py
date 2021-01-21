import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import cv2
import os
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone


def video_to_frame(videos_src_path, videos_save_path):

    if not os.path.exists(videos_save_path):
        os.makedirs(videos_save_path)

    videos = os.listdir(videos_src_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    videos = filter(lambda x: x.endswith('mp4'), videos)  # 将mp4文件读进来，可改为avi等格式
    videos = tqdm(videos, desc=f'video_to_frame')
    for each_video in videos:
        frame_count = 1
        # 得到每个文件夹的名字, 并指定每一帧的保存路径
        each_video_name, _ = each_video.split('.')
        # if not os.path.exists(videos_save_path + '/' + each_video_name):
        #     os.mkdir(videos_save_path + '/' + each_video_name)
        each_video_save_full_path = videos_save_path + '/'
        # 得到完整的视频路径
        each_video_full_path = os.path.join(videos_src_path, each_video)
        # 用OpenCV一帧一帧读取出来
        cap = cv2.VideoCapture(each_video_full_path)
        success = True
        while (success):
            success, frame = cap.read()
            # print('Read a new frame: ', success)
            # params = []
            # params.append(1)

            if frame is not None and frame_count % 1 == 0:  # 每？帧取一帧图片保存下来，可以自己修改
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame)
                # print(frame_count)
            frame_count = frame_count + 1
        cap.release()

def gen_new_data(lines, input_shape, path):
    lines = tqdm(lines, desc=f'gen_new_data')
    for t in lines:
        # annotation_line = lines[t]
        # line = annotation_line.split()
        image = cv2.imread(t)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ih, iw, _ = image.shape
        w, h = input_shape
        resize_ratio = min(w / iw, h / ih)
        resize_w = int(iw * resize_ratio)
        resize_h = int(ih * resize_ratio)
        image_resize = cv2.resize(image, (resize_w, resize_h))

        image_paded = np.full((h, w, 3), 128)
        dw = int((w - resize_w) / 2)
        dh = int((h - resize_h) / 2)
        image_paded[dh: resize_h + dh, dw: resize_w + dw, :] = image_resize

        new_image0 = image_paded[:int(h/2+52), :int(w/2+52),:]
        new_image1 = image_paded[:int(h/2+52), int(w/2-52):,:]
        new_image2 = image_paded[int(h/2-52):, :int(w/2+52):,:]
        new_image3 = image_paded[int(h/2-52):, int(w/2-52):,:]
        new_image=(new_image0,new_image1,new_image2, new_image3)

        if not os.path.exists(path):
            os.makedirs(path)

        for j in range(4):
            box_clip_filepath = os.path.join(path, t.split("\\")[-1].split('.')[0]+ "_%d"%j + ".jpg")
            cv2.imwrite(box_clip_filepath, new_image[j])

class TestDataset(Dataset):
    def __init__(self, lines):
        super(TestDataset, self).__init__()
        self.test_lines = lines
        self.test_nums = len(lines)

    def __len__(self):
        return self.test_nums

    def __getitem__(self, index):
        line = self.test_lines[index]
        # line = one_line.split()
        image_src = cv2.imread(line)
        h, w, _ = image_src.shape
        image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

        img = np.array(image, dtype=np.float32)
        img = np.transpose(img / 255.0, (2, 0, 1))

        return img, [h, w, line]

def test_dataset_collate(batch):
    # images = []
    inputs = []
    targets = []
    shapes = []

    for img, infos in batch:
        # images.append(img_src)
        inputs.append(img)
        # targets.append(labels)
        shapes.append(infos)

    inputs = np.array(inputs, dtype=np.float32)

    return inputs, shapes

# def create_model(device):
#
#     train_weights = "./save_weights1/lala.pth"
#     assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
#     model = torch.load(train_weights, map_location=device)
#     model['model'].to(device)
#     # model[].cuda()
#     return model

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

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

def box_area(boxes):

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])  # right-bottom [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou

def decode_bboxes(outputs, infos, df):


    #1
    for j in range(len(infos)):
        mask = list(infos[j][2].split('_')[-1])[0]
        if mask == '1':
            outputs[j]['boxes'][:, [0, 2]] = 992 - 52 + outputs[j]['boxes'][:, [0, 2]]
        elif mask == '2':
            outputs[j]['boxes'][:, [1, 3]] = 992 - 52 + outputs[j]['boxes'][:, [1, 3]]
        elif mask == '3':
            outputs[j]['boxes'][:, [1, 3]] = 992 - 52 + outputs[j]['boxes'][:, [1, 3]]
            outputs[j]['boxes'][:, [0, 2]] = 992 - 52 + outputs[j]['boxes'][:, [0, 2]]
        # 2
        ih, iw = 720, 1280
        w, h = 1880, 1880
        resize_ratio = min(w / iw, h / ih)
        resize_w = int(iw * resize_ratio)
        resize_h = int(ih * resize_ratio)
        dw = int((w - resize_w) / 2)
        dh = int((h - resize_h) / 2)

        outputs[j]['boxes'][:, [0, 2]] = (outputs[j]['boxes'][:, [0, 2]] -dw) /resize_ratio
        outputs[j]['boxes'][:, [1, 3]] = (outputs[j]['boxes'][:, [1, 3]] -dh) /resize_ratio


        gameKey = str(infos[j][2].split("\\")[-1].split("_")[0])
        playID = str(infos[j][2].split("\\")[-1].split("_")[1])
        view = str(infos[j][2].split("\\")[-1].split("_")[2])
        frame = str(infos[j][2].split("\\")[-1].split("_")[3])
        video = gameKey + '_' + playID + '_' + view + '.mp4'
        for i in range(len(outputs[j]['boxes'])):
            left = outputs[j]['boxes'][i][0]
            width = outputs[j]['boxes'][i][2] - outputs[j]['boxes'][i][0]
            top = outputs[j]['boxes'][i][1]
            height = outputs[j]['boxes'][i][3] - outputs[j]['boxes'][i][1]
            df = df.append(pd.DataFrame({'gameKey':gameKey, 'playID':[int(playID)], 'view':view, 'video':video, 'frame':[frame],
                                        'left':[int(left)], 'width':[int(width)], 'top':[int(top)], 'height':[int(height)]}))

    return df

def calculate_iou(test_data):

    test_data['name'] = test_data['video'] + test_data['frame']
    test_data['xmin'] = test_data['left']
    test_data['ymin'] = test_data['top']
    test_data['xmax'] = test_data['left'] + test_data['width']
    test_data['ymax'] = test_data['top'] + test_data['height']
    test_data['iou'] = 0

    test_data[['xmin','ymin','xmax','ymax']] = test_data[['xmin','ymin','xmax','ymax']].astype('float')


    names_mat = np.empty(shape=(0, 5))
    names = test_data['name'].unique()
    names = tqdm(names)
    for i, name in enumerate(names):
        name_mat = test_data.loc[test_data['name'] == name][["xmin", 'ymin', 'xmax', 'ymax', 'iou']].values
        name_mat_copy = name_mat.copy()
        iou = box_iou(name_mat, name_mat_copy)
        for j in range(iou.shape[0]):
            iou_seat = np.where((iou[j, :] > 0) & (iou[j, :] < 0.9))
            if len(iou_seat[0]) == 0:
                continue
            iou_value = iou[j, iou_seat]
            name_mat[iou_seat, 4] = iou_value
        names_mat = np.concatenate((names_mat, name_mat), axis=0)

    test_data.loc[:, 'iou'] = names_mat[:, 4]  #这里数值可能不匹配

    return test_data


@torch.no_grad()
def main(split_pictures, device, pth_path):
    cpu_device = torch.device("cpu")

    # create model
    model = create_model(num_classes=2)
    model = load_model_pth(model, pth_path)
    model.to(device)
    model.eval()
    # read test_data

    test_data_set = TestDataset(split_pictures)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    test_data_loader = torch.utils.data.DataLoader(test_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=nw,
                                                    drop_last=False,
                                                    collate_fn=test_dataset_collate)

    test_data_loader = tqdm(test_data_loader)
    df = pd.DataFrame(columns=('gameKey', 'playID', 'view', 'video', 'frame', 'left', 'width', 'top', 'height'))

    for i, batch in enumerate(test_data_loader):
        images, infos = batch[0], batch[1]
        images = torch.from_numpy(images).type(torch.FloatTensor).to(device)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        outputs = model(images)
        outputs = [{k: v.to(cpu_device).numpy() for k, v in t.items()} for t in outputs]

        df = decode_bboxes(outputs, infos, df)

    df_iou = calculate_iou(df)

    #筛选出iou重合的行
    final_df = df_iou.loc[df_iou['iou']>0]
    final_df = final_df.drop(['name', "xmin", 'ymin', 'xmax', 'ymax', 'iou'], axis=1)
    return final_df




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    videos_src_path = r'E:\kaggle\nfl-impact-detection\test'  # 提取图片的视频文件夹
    videos_save_path = r'E:\kaggle\nfl-impact-detection\test_picture'  # 保存图片的路径
    # video_to_frame(videos_src_path, videos_save_path)

    test_data_dir = r'E:\kaggle\nfl-impact-detection\test_picture'
    pictures = [os.path.join(test_data_dir, i) for i in os.listdir(test_data_dir) if i.endswith("jpg")]
    test_data_split_dir = r"E:\kaggle\nfl-impact-detection\test_clip_images_992"
    # gen_new_data(pictures, (1880, 1880), test_data_split_dir)
    split_pictures = [os.path.join(test_data_split_dir, i) for i in os.listdir(test_data_split_dir) if i.endswith("jpg")]

    pth_path = './save_weights/Epoch_022_Loss_0.0736_lr_0.000059.pth'

    final_df =  main(split_pictures, device, pth_path)
    final_df.to_csv(r"E:\kaggle\nfl-impact-detection\final_df.csv", index=False)

