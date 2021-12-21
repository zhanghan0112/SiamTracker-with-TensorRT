import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round
from lib.models.lighttrack_speed_trt import LightTrackTRT
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import threading

class LightTrack_tem(object):
    def __init__(self, info, even=0):
        super(LightTrack_tem, self).__init__()
        self.info = info  # model and benchmark info
        self.stride = info.stride
        self.even = even
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def normalize(self, x):
        """ input is in (C,H,W) format"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x

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

    def init(self, im, target_pos, target_sz, model):
        state = dict()

        p = Config(stride=self.stride, even=self.even)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # load hyper-parameters from the yaml file
        prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
        if len(prefix) == 0:
            prefix = [self.info.dataset]
        yaml_path = os.path.join(os.path.dirname(__file__), '../../experiments/test/VOT/', 'LightTrack.yaml')
        cfg = load_yaml(yaml_path)
        cfg_benchmark = cfg[self.info.dataset]
        p.update(cfg_benchmark)
        p.renew()
        p.instance_size = cfg_benchmark['small_sz']
        p.renew()

        self.grids(p)  # self.grid_to_search_x, self.grid_to_search_y

        net = model
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        #####search area########
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        self.zf = net.template(z.cuda())
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))
        else:
            raise ValueError("Unsupported window type")
        state['p'] = p
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['zf'] = self.zf
        return state

class Lighttrack_track(object):
    def __init__(self):
        super(Lighttrack_track, self).__init__()
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        engine_path = 'model_track.trt'
        with open(engine_path,"rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size,dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def normalize(self, x):
        """ input is in (C,H,W) format"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x

    def track(self, state, im):
        threading.Thread.__init__(self)
        self.ctx.push()
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        p = state['p']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']
        zf = state['zf']

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
        x_crop = x_crop.cpu().numpy().astype(np.float32)
        zf = zf.detach().cpu().numpy().astype(np.float32)
        np.copyto(host_inputs[0],np.ascontiguousarray(zf).ravel())
        np.copyto(host_inputs[1],np.ascontiguousarray(x_crop).ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        cuda.memcpy_htod_async(cuda_inputs[1], host_inputs[1], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings,stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        stream.synchronize()
        self.ctx.pop()
        cls_score = host_outputs[0]
        bbox_pred = host_outputs[1]
        cls_score = torch.from_numpy(cls_score).reshape(1,1,16,16)
        bbox_pred = torch.from_numpy(bbox_pred).reshape(1,4,16,16)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()
        self.grids(p)

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
        state['cls_score'] = cls_score
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
