import os
import os.path as osp
import numpy as np
import cv2
import paddlex as pdx
from visualdl import LogWriter

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch_name', type=str, default=None,
                        help='想要查看的模型轮次名称')
    parser.add_argument('--pic_num', type=int, default=1,
                        help='每个模型预测的图像数量')
    parser.add_argument('--model_save_dir', type=str, default='output/deeplabv3p_mobilenetv3_large_ssld',
                        help='模型保存的路径')                                       
    return parser.parse_args()


def get_img_paths():
    data_list = './dataset/val_list.txt'
    pic_names = open(data_list, 'r').readlines()
    np.random.shuffle(pic_names)
    img_paths = []
    for i in range(args.pic_num):
        pic_name = pic_names[i].strip().split(' ')[0]
        img_paths.append(osp.join('dataset', pic_name))
    return img_paths

def main(epoch_name_list, out_dir):
    if len(epoch_name_list) > 10:
        print('每个窗口最多仅支持展示十张图片！最终结果会随机抽样！')
    vdl_writer = LogWriter(out_dir)
    
    img_paths = get_img_paths()
    #print(img_paths)

    for id, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        tag = '0. OriginalImage/' + str(id)
        vdl_writer.add_image(
            tag=tag, img=img, step=0)
    for epoch_name in epoch_name_list:
        model_name = 'epoch_' + epoch_name
        model = pdx.load_model(osp.join(args.model_save_dir, model_name))
        for id, img_path in enumerate(img_paths):
            tag = 'Prediction/{}'.format(str(id))
            result = model.predict(img_path)
            vis_res = pdx.seg.visualize(img_path, result, 0., save_dir=None)
            vdl_writer.add_image(
                tag=tag, img=vis_res, step=int(epoch_name))
    print('执行完毕，请设置logdir后，开启服务进行查看')

    
if __name__ == '__main__':
    args = parse_args()
    out_dir = osp.join(args.model_save_dir, 'vis_predict_log/')
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    epoch_name = args.epoch_name.strip()
    epoch_name_list = epoch_name.split(" ")
    main(epoch_name_list, out_dir)






