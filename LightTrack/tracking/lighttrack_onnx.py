import sys
import os
import _init_paths
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import time
import onnx
import onnxruntime
import numpy as np
from lib.models.models import LightTrackM_Speed, LightTrackM_Supernet
from lib.models.super_model_DP import Super_model_DP_retrain
from lib.utils.utils import load_pretrain
from lib.utils.transform import name2path
from lib.models.backbone import build_subnet
from lib.models.submodels import build_subnet_head, build_subnet_BN, build_subnet_feat_fusor

def getdata(b=1,x=256,z=8):
    x = torch.randn(1,3,x,x)
    z = torch.randn(1,96,z,z)
    return x, z

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class LightTrackM_Subnet(Super_model_DP_retrain):
    def __init__(self, path_name, search_size=288, template_size=128, stride=16, adj_channel=128):
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size,stride=stride)
        model_cfg = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                         stride=stride, adj_channel=adj_channel, build_module=False)
        path_backbone, path_head, path_ops = name2path(path_name, sta_num=model_cfg.sta_num)
        # build the backbone
        self.features = build_subnet(path_backbone, ops=path_ops)  # sta_num is based on previous flops
        # build the neck layer
        self.neck = build_subnet_BN(path_ops, model_cfg)
        # build the Correlation layer and channel adjustment layer
        self.feature_fusor = build_subnet_feat_fusor(path_ops, model_cfg, matrix=True, adj_channel=adj_channel)
        # build the head
        self.head = build_subnet_head(path_head, channel_list=model_cfg.channel_head, kernel_list=model_cfg.kernel_head,
                                      inchannels=adj_channel, linear_reg=True, towernum=model_cfg.tower_num)
    
    def forward(self, zf, search):
        xf = self.features(search)
        zf, xf = self.neck(zf,xf)
        feat_dict = self.feature_fusor(zf, xf)
        oup = self.head(feat_dict)
        return oup

def convert_tracking_model(net):
    search = torch.randn(1, 3, 256, 256).cuda()
    zf = torch.randn(1,96,8,8).cuda()
    ort_inputs = {'zf': to_numpy(zf).astype(np.float32),
                  'search': to_numpy(search).astype(np.float32)}
    ###########complete model pytorch->onnx#############
    print("Converting tracking model now!")
    torch.onnx.export(net, (zf,search), 'model_forward.onnx', export_params=True,
          opset_version=11, do_constant_folding=True, input_names=['zf','search'],
          output_names=['cls','reg'])
    #######load the converted model and inference#######
    with torch.no_grad():
        oup = net(zf, search)
        onnx_model = onnx.load("model_forward.onnx")
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession("model_forward.onnx")
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(oup['cls']), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(oup['cls'])-ort_outs[0]))))
        print("The deviation between the second output: {}".format(np.max(np.abs(to_numpy(oup['reg'])-ort_outs[1]))))
    print(onnxruntime.get_device())
    print("Tracking onnx model has done!")

def convert_template_model(net):
    template = torch.randn(1, 3, 127, 127).cuda()
    onnx_inputs = {'template': to_numpy(template).astype(np.float32)}
    backbone = net.features
    print("Converting template model now!")
    torch.onnx.export(backbone, (template), 'model_backbone.onnx', export_params=True,
          opset_version=11, do_constant_folding=True, input_names=['template'],
          output_names=['out'])
    with torch.no_grad():
        out = backbone(template)
        ##len(out)=4 len(pos)=4
        onnx_backbone = onnx.load("model_backbone.onnx")
        onnx.checker.check_model(onnx_backbone)
        ort_session = onnxruntime.InferenceSession("model_backbone.onnx")
        ort_outs = ort_session.run(None, onnx_inputs)
        #len(ort_outs) = 8 where ort_outs[4] == pos[0],ort_outs[0] == out[0]
        np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(out)-ort_outs[0]))))
    print(onnxruntime.get_device())
    print("Template onnx model has done!")

def inference_speed_track(net):
    T_w = 100  # warmup
    T_t = 1000  # test
    torcht = 0
    onnxt = 0
    ort_session = onnxruntime.InferenceSession("model_forward.onnx")
    with torch.no_grad():
        x, z = getdata()
        x_cuda, z_cuda = x.cuda(),z.cuda()
        ort_inputs = {'zf': to_numpy(z_cuda).astype(np.float32),
                      'search': to_numpy(x_cuda).astype(np.float32)}
        for i in range(T_w):
            oup = net(z_cuda, x_cuda)
            nxp = ort_session.run(None, ort_inputs)
        for i in range(T_t):
            torch_s = time.time()
            oup = net(z_cuda, x_cuda)
            torch_e = time.time() - torch_s
            onnx_s = time.time()
            nxp = ort_session.run(None, ort_inputs)
            onnx_e = time.time() - onnx_s
            torch_se = torch_e - torch_s
            onnx_se = onnx_e - onnx_s
            torcht += torch_se
            onnxt += onnx_se
    print('The tracking process inference speed of pytorch model: %.2f FPS' % (T_w/torch_e))
    print('The tracking process inference speed of onnx model: %.2f FPS' % (T_t/onnx_e))

if __name__ == "__main__":
    # test the running speed
    path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'  # our 530M model
    use_gpu = True
    model_path = 'snapshot/LightTrackM/LightTrackM.pth'
    net = LightTrackM_Subnet(path_name=path_name)
    model = load_pretrain(net,model_path)
    if use_gpu:
        model.cuda()
        model.eval()
    ######convert and check tracking pytorch model to onnx#####
    convert_tracking_model(model)
    ######convert and check template pytorch model to onnx#####
    convert_template_model(model) 
    ######test tracking process inference speed#####
    inference_speed_track(model)
