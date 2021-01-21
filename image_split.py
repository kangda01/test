'''数据处理，将低分辨率的图片转换成高分辨率的图片，
再将高分辨率的图片分成四份，并将box做对应处理'''


import numpy as np
# from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt



def gen_new_data(lines, input_shape):
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


        path = "E:\\kaggle\\nfl-impact-detection\\test_clip_images_992"
        if not os.path.exists(path):
            os.makedirs(path)

        for j in range(4):
            box_clip_filepath = os.path.join(path, t.split("\\")[-1].split('.')[0]+ "_%d"%j + ".jpg")
            cv2.imwrite(box_clip_filepath, new_image[j])

if __name__ == '__main__':
    test_data_dir = r'E:\kaggle\nfl-impact-detection\test_picture'
    pictures = [os.path.join(test_data_dir, i) for i in os.listdir(test_data_dir) if i.endswith("jpg")]
    # pictures = filter(lambda x: x.endswith('jpg'), pictures)
    gen_new_data(pictures, (1880,1880))

