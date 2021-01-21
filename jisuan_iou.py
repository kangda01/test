import pandas as pd
import numpy as np
from tqdm import tqdm
"""计算每张图片中的box与其它框的iou"""

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])  # right-bottom [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


train_data = pd.read_csv(r"E:\kaggle\nfl-impact-detection\train_labels1.csv")

#合并几列的内容
train_data['gameKey'] = train_data['gameKey'].astype('str')
train_data['playID'] = train_data['playID'].astype('str')
train_data['frame'] = train_data['frame'].astype('str')
train_data['name'] = train_data['gameKey'] + train_data['playID'] + train_data['view'] + train_data['video'] + train_data['frame']

names_mat = np.empty(shape=(0,5))
names = train_data['name'].unique()
names = tqdm(names)
for i, name in enumerate(names):
    name_mat = train_data.loc[train_data['name'] == name][["xmin",'ymin','xmax', 'ymax', 'iou']].values
    name_mat_copy = name_mat.copy()
    iou = box_iou(name_mat, name_mat_copy)
    for j in range(iou.shape[0]):
        iou_seat = np.where((iou[j,:]>0) & (iou[j,:]<1))
        if len(iou_seat[0]) ==0:
            continue
        iou_value = iou[j,iou_seat]
        name_mat[iou_seat,4] = iou_value
    names_mat = np.concatenate((names_mat,name_mat), axis=0)

# df = pd.DataFrame(names_mat)
# tra = train_data.copy()
# tra['iou'].iloc[0:200] = names_mat[:,4]
train_data.loc[:,'iou'] = names_mat[:,4]
train_data.to_csv(r"E:\kaggle\nfl-impact-detection\train_labels1.csv", index=False)



# lala = train_data.loc[train_data['name'] == name[0]][["xmin",'ymin','xmax', 'ymax']]
# lala = train_data.groupby('name')
# train_data.dtypes
# box1 = np.array([[586,	301,	607,	327]])
# box2 = np.array([[568,	326,	591,	352
# ]])
#
#
#
# a = box_iou(box1,box2)