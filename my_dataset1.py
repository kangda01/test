from random import shuffle
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2

class TrainDataset(Dataset):
    def __init__(self, train_lines, image_size, mosaic=False):
        super(TrainDataset, self).__init__()

        self.train_lines = train_lines
        self.train_nums = len(train_lines)
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True

    def __len__(self):
        return self.train_nums

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a



    def get_random_data(self, annotation_line, input_shape):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        # image = Image.open(line[0])
        image = cv2.imread(line[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        if len(box) == 0:
            return image, []
        return image, box




    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_nums
        index = index % n
        if self.mosaic:
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
            else:
                img, y = self.get_random_data(lines[index], self.image_size[0:2])
            self.flag = bool(1-self.flag)
        else:
            img, y = self.get_random_data(lines[index], self.image_size[0:2])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            # boxes[:, 0] = boxes[:, 0] / self.image_size[0]
            # boxes[:, 1] = boxes[:, 1] / self.image_size[1]
            # boxes[:, 2] = boxes[:, 2] / self.image_size[0]
            # boxes[:, 3] = boxes[:, 3] / self.image_size[1]
            #
            # boxes = np.maximum(np.minimum(boxes, 1), 0)
            # boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            # boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            #
            # boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            # boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

            labels = np.array(y[:, 4], dtype=np.int64)
            # y = np.concatenate([boxes, y[:, -1:]], axis=-1)


        image = np.array(img, dtype=np.float32)

        image = np.transpose(image / 255.0, (2, 0, 1))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # tmp_inp = np.transpose(img, (2, 0, 1))
        # tmp_targets = np.array(y, dtype=np.float32)
        return image, target


class TestDataset(Dataset):
    def __init__(self, lines, image_size):
        super(TestDataset, self).__init__()
        self.test_lines = lines
        self.test_nums = len(lines)
        self.image_size = image_size

    def __len__(self):
        return self.test_nums

    def __getitem__(self, index):
        one_line = self.test_lines[index]
        line = one_line.split()
        image_src = cv2.imread(line[0])
        h, w, _ = image_src.shape
        image = cv2.resize(image_src, (self.image_size[1], self.image_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        y = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

        img = np.array(image, dtype=np.float32)
        img = np.transpose(img / 255.0, (2, 0, 1))

        if len(y) != 0:
            boxes = np.array(y[:, :4], dtype=np.float32)
            labels = np.array(y[:, 4], dtype=np.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return image_src, img, target, [h, w, line[0]]


# @staticmethod
# def train_dataset_collate(batch):
#     return tuple(zip(*batch))

# DataLoader中collate_fn使用
def train_dataset_collate(batch):
    images = []
    bboxes = []
    gtboxex_counts = []
    # batch_boxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    #下面这些是将每个批次中的boxes也组合成批次，方面后续用矩阵运算替代for循环。
    # for k, y in enumerate(bboxes):
    #     gtboxex_counts.append(y.shape[0])
    # each_batch_max_gtboxex_count = max(gtboxex_counts)
    # for k, y in enumerate(bboxes):
    #     pad_row = each_batch_max_gtboxex_count - y.shape[0]
    #     bboxes[k] = np.pad(y, ((0, pad_row), (0, 0)), 'constant')
    #     # 把多添加的lables变成-1
    #     bboxes[k][y.shape[0]:each_batch_max_gtboxex_count, 4] = -1
    #
    images = np.array(images)
    # bboxes = np.array(bboxes)
    return images, bboxes

def test_dataset_collate(batch):
    srcs = []
    inputs = []
    targets = []
    shapes = []

    for img_src, img, labels, infos in batch:
        srcs.append(img_src)
        inputs.append(img)
        targets.append(labels)
        shapes.append(infos)

    inputs = np.array(inputs, dtype=np.float32)

    return srcs, inputs, targets, shapes