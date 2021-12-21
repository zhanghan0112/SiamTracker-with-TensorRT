import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import onnx
import onnxruntime
import time

def getdata(b=1,x=256,z=8):
    x = torch.randn(3,256,256)
    z = torch.randn(96,8,8)
    return x, z

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n"+str(self.host)+"\nDevice:\n"+str(self.device)
    def __repr__(self):
        return self.__str__()

def do_inference_v2(context, bindings, inputs, outputs, stream):
    #for TensorRT 7.0+
    [cuda.memcpy_htod_async(inp.device,inp.host,stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host,out.device,stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class LightTrack_TRT(object):
    def _load_engine(self):
        engine_path = 'model_track.trt'
        with open(engine_path,"rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def __init__(self):
        self.inference_fn = do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def allocate_buffers(self,engine):
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

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def track(self):
        for i in range(100):
            x = np.random.randn(1,3,256,256).astype(np.float32)
            zf = np.random.randn(1,96,8,8).astype(np.float32)
            self.inputs[0].host = np.ascontiguousarray(zf)
            self.inputs[1].host = np.ascontiguousarray(x)
            trt_outputs = self.inference_fn(context=self.context,
                    bindings=self.bindings,
                    inputs = self.inputs,
                    outputs = self.outputs,
                    stream = self.stream)
        e=0
        for i in range():
            x = np.random.randn(1,3,256,256).astype(np.float32)
            zf = np.random.randn(1,96,8,8).astype(np.float32)
            self.inputs[0].host = np.ascontiguousarray(zf)
            self.inputs[1].host = np.ascontiguousarray(x)
            s = time.time()
            trt_outputs = self.inference_fn(context=self.context,
                    bindings=self.bindings,
                    inputs = self.inputs,
                    outputs = self.outputs,
                    stream = self.stream)
            e += time.time() - s
        print("FPS is : ", 1000/e) 
        return trt_outputs[0]

if __name__ == '__main__':
    trt_out = LightTrack_TRT().track()
    #print(trt_out)
    
