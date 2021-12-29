import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import cv2
import yaml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import threading
from easydict import EasyDict as edict
import onnxruntime
import torch
import torch.nn.functional as F
from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round
from lib.models.lighttrack_speed_trt import LightTrackTRT
import lib.models.models as models
from dataset import DatasetFactory
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round

class Lighttrack(object):
    def __init__(self, info, even=0):
        super(Lighttrack, self).__init__()
        self.info = info  # model and benchmark info
        self.stride = info.stride
        self.even = even
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.ort_session_z = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), '../model_backbone.onnx)) #load the onnx model of backone for template feature.

    def normalize(self, x):
        """ input is in (C,H,W) format"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x
  
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def init(self, im, target_pos, target_sz):
        state = dict()

        p = Config(stride=self.stride, even=self.even)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # load hyper-parameters from the yaml file
        prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
        if len(prefix) == 0:
            prefix = [self.info.dataset]
        yaml_path = os.path.join(os.path.dirname(__file__), '../experiments/test/VOT/', 'LightTrack.yaml') 
        cfg = load_yaml(yaml_path)  #modify the dataset name of yaml if you want to test other datasets
        cfg_benchmark = cfg[self.info.dataset]
        p.update(cfg_benchmark)
        p.renew()

        # if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        #     p.instance_size = cfg_benchmark['big_sz']
        #     p.renew()
        # else:  #modify fixed size of input image
        p.instance_size = cfg_benchmark['small_sz']
        p.renew()

        self.grids(p)  # self.grid_to_search_x, self.grid_to_search_y

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        ort_inputs_z = {'template': self.to_numpy(z).astype(np.float32)}
        self.ort_outs = self.ort_session_z.run(None, ort_inputs_z)
        self.zf = self.ort_outs[0]                                                               

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))
        else:
            raise ValueError("Unsupported window type")

        state['p'] = p
        # state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p, debug=False):
        cls_score, bbox_pred = net.track(x_crops, self.zf.astype(np.float32)
        # print(cls_score)
        # cls_score, bbox_pred = net.track(x_crops)
        cls_score = torch.from_numpy(cls_score).reshape(1,1,16,16)
        bbox_pred = torch.from_numpy(bbox_pred).reshape(1,4,16,16)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])
        if debug:
            return target_pos, target_sz, cls_score[r_max, c_max], cls_score
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, state, im, net):
        p = state['p']
        # net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        state['x_crop'] = x_crop.clone()  # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)
        x_crop = x_crop.numpy().astype(np.float32)
        debug = True
        if debug:
            target_pos, target_sz, _, cls_score = self.update(net, x_crop, target_pos, target_sz * scale_z,
                                                              window, scale_z, p, debug=debug)
            state['cls_score'] = cls_score
        else:
            target_pos, target_sz, _ = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z,
                                                   window, scale_z, p, debug=debug)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p
        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        # print('ATTENTION',p.instance_size,p.score_size)
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class Config(object):
    def __init__(self, stride=8, even=0):
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.windowing = 'cosine'
        if even:
            self.exemplar_size = 128
            self.instance_size = 256
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        # total_stride = 8
        # score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 0.94

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        # self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
        self.score_size = int(round(self.instance_size / self.total_stride))

class inferThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # self.net = LightTrackTRT()

    def run(self,args):
        dataset_root = ''
        siam_info = edict()
        siam_info.arch = args.arch
        siam_info.dataset = args.dataset
        siam_info.stride = args.stride

        # create model
        tracker = Lighttrack(siam_info, even=args.even)
        net = LightTrackTRT()
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
                    outputs = tracker.track(state, img, net)
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
                # net.destory()
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
        net.destory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lghttrack tracking')
    parser.add_argument('--dataset', type=str, default='OTB',help='datasets')
    parser.add_argument('--video', default='', type=str,help='eval one special video')
    parser.add_argument('--name', default='lighttrack-otb', type=str,help='name of results')
    parser.add_argument('--arch', dest='arch', help='backbone architecture')
    parser.add_argument('--stride', type=int, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='NULL')
    args = parser.parse_args()  
    thread1 = inferThread()
    thread1.run(args)
