import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
import onnx
import onnxruntime
import time

def getdata(b=1,x=256,z=8):
    x = torch.randn(1,3,256,256)
    z = torch.randn(1,96,8,8)
    return x, z
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n"+str(self.host)+"\nDevice:\n"+str(self.device)
    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs=[]
    outputs=[]
    bindings=[]
    stream=cuda.Stream()
    for binding in engine:
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            size=trt.volume(binding_dims)
        elif len(binding_dims) == 3:
            size = trt.volume(binding_dims) * engine.max_batch_size
        else:
            raise ValueError('bad dims of binging %s: %s' %(binding, str(binding_dim)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size,dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem,device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem,device_mem))
    return inputs,outputs,bindings,stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    #for TensorRT 7.0+
    [cuda.memcpy_htod_async(inp.device,inp.host,stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host,out.device,stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

def postprocess_the_outputs(outputs, shape_of_output):
    outputs = outputs.reshape(*shape_of_output)
    return outputs 

class LightTrackTRT(object):
    
    def _load_engine(self):
        engine_path = 'model_track_int8.trt'
        with open(engine_path,"rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def __init__(self):
        self.ctx = cuda.Device(0).make_context()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def track(self, x, zf):
        # threading.Thread.__init__(self)
        self.ctx.push()
        #restore
        stream = self.stream
        context = self.context
        engine = self.engine
        inputs = self.inputs
        outputs = self.outputs
        bindings = self.bindings
        # x = np.random.randn(1,3,256,256).astype(np.float32)
        # zf = np.random.randn(1,96,8,8).astype(np.float32)
        inputs[0].host = np.ascontiguousarray(zf)
        inputs[1].host = np.ascontiguousarray(x)
        trt_outputs = do_inference_v2(context=context,
                bindings=bindings,
                inputs = inputs,
                outputs = outputs,
                stream = stream)
        self.ctx.pop()
        
        return trt_outputs[0], trt_outputs[1]

    def destory(self):
        self.ctx.pop()

# class LightTrack_TRT(object):
#     def _load_engine(self):
#         engine_path = 'sample.engine'
#         with open(engine_path,"rb") as f, trt.Runtime(self.trt_logger) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#     def __init__(self):
#         self.inference_fn = do_inference_v2
#         self.trt_logger = trt.Logger(trt.Logger.INFO)
#         self.engine = self._load_engine()
#         #self.input_shape = get_input_shape(self.engine)

#         try:
#             self.context = self.engine.create_execution_context()
#             self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
#         except Exception as e:
#             raise RuntimeError('fail to allocate CUDA resources') from e

#     def track(self, x, zf):
#         # x = np.random.randn(1,3,256,256).astype(np.float32)
#         # zf = np.random.randn(1,96,8,8).astype(np.float32)
#         self.inputs[0].host = np.ascontiguousarray(x)
#         self.inputs[1].host = np.ascontiguousarray(zf)
#         trt_outputs = self.inference_fn(context=self.context,
#                 bindings=self.bindings,
#                 inputs = self.inputs,
#                 outputs = self.outputs,
#                 stream = self.stream)
#         return trt_outputs[0], trt_outputs[1]
