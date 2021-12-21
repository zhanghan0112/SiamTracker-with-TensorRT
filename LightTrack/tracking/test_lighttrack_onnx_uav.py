# Copyright (c) SenseTime. All Rights Reserved.
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import cv2
import torch
import numpy as np
from easydict import EasyDict as edict
from lib.tracker.lighttrack_onnx import Lighttrack_onnx
import lib.models.models as models
from dataset import DatasetFactory
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

parser = argparse.ArgumentParser(description='lghttrack tracking')
parser.add_argument('--dataset', type=str, default='OTB',
        help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--name', default='lighttrack-onnx', type=str,
        help='name of results')
parser.add_argument('--arch', dest='arch', help='backbone architecture')
parser.add_argument('--stride', type=int, help='network stride')
parser.add_argument('--even', type=int, default=0)
parser.add_argument('--resume', type=str, help='pretrained model')
parser.add_argument('--path_name', type=str, default='NULL')

args = parser.parse_args()
def main():
    dataset_root = ''
    net_path = ''
    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.stride = args.stride

    # create model
    tracker = Lighttrack_onnx(siam_info, even=args.even)
    # if args.path_name != 'NULL':
    #     siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    # else:
    #     siam_net = models.__dict__[args.arch](stride=siam_info.stride)

    # print('===> init Siamese <====')

    # siam_net = load_pretrain(siam_net, args.resume)
    # siam_net.eval()
    # siam_net = siam_net.cuda()
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            #convert bgr to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                state = tracker.init(img, target_pos, target_sz)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(state, img)
                pos = state['target_pos']
                sz = state['target_sz']
                pred_bbox = [float(max(float(0), pos[0] - sz[0] / 2)), float(max(float(0), pos[1] - sz[1] / 2)), float(sz[0]),
                            float(sz[1])]
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
        toc /= cv2.getTickFrequency()
        # save results
        if 'GOT-10k' == args.dataset:
            video_path = os.path.join('results', args.dataset, args.name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            result_path = os.path.join(video_path,'{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
        else:
            model_path = os.path.join('results', args.dataset, args.name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
