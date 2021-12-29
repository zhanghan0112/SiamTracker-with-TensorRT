# SiamTracker-with-TensorRT
The modify PyTorch version of Siam-trackers which are speed-up by TensorRT or ONNX. **[Updating...]**
</br> Examples demonstrating how to optimize PyTorch models with ONNX or TensorRT and do inference on NVIDIA Jetson platforms. 
   + All reported speeds contain pre-process process.

| Dataset | Tracker | Platform | Origin Success(%)/Speed(fps) | TensorRT-FP32 Success(%)/Speed(fps) | TensorRT-FP16 Success(%)/Speed(fps) | ONNX Success(%)/Speed(fps) |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| UAV123 | LightTrack | Jetson AGX Xavier | 62.6/15~ | 61.7/38~ | 61.3/42~ | 61.6/36~ |
   + For fixed image size of 256x256, the performance of ONNX and TensorRT model is not equal to PyTorch model which input size is either of 288x288 or 256x256. If the input size is same, they are equal.    
## Installation
### Prerequisites
The code in this repository was tested on Jetson AGX Xavier DevKits. In order to go futher, 
first make sure you have the proper version of image (JetPack) installed on the target Jetson system. 
If not, you can refer to this: [Set up Jetson Xavier family](https://blog.csdn.net/xiao_zhang99/article/details/121704925?spm=1001.2014.3001.5501) , we use JetPack4.6 in this repo. 
  + More specifically, the target Jetson system must have ONNX and TensorRT libraries installed.
Check which version of TensorRT has been installed on your Jetson system:
    ```python
    sudo -H pip3 install jetson-stats
    jetson_release -v
    ```
  + Install ONNX
    ```python
    sudo apt-get install protobuf-compiler libprotoc-dev
    pip3 install onnx
    ```
  + Install ONNX Runtime. [Jetson_Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) has provided pre-build version on Jetson systems, you can download proper version on this website, more details refer to [Set up Jetson Xavier family](https://blog.csdn.net/xiao_zhang99/article/details/121704925?spm=1001.2014.3001.5501).
    ```python
    # Download pip wheel from location mentioned above
    $ wget https://nvidia.box.com/s/bfs688apyvor4eo8sf3y1oqtnarwafww -O onnxruntime_gpu-1.6.0-cp36-cp36m-linux_aarch64.whl
    # Install pip wheel
    $ pip3 install onnxruntime_gpu-1.6.0-cp36-cp36m-linux_aarch64.whl
    ```
  + Install pycuda
    `pip3 install pycuda`
## Convert PyTorch Model to ONNX
  + In LightTrack repo, first modify the loaded PyTorch model path in `LightTrack/tracking/lighttrack_onnx.py`. To obtain ONNX model:
    ```python
    python3 tracking/lighttrack_onnx.py
    ```
## Build engine
  + If the TensorRT libraries has been installed, `trtexec` exists in `/usr/src/tensorrt/bin`, then run:
    ```python
    /usr/src/tensorrt/bin/trtexec --onnx=/path/to/your/onnx/model --saveEngine=LightTrack.trt --fp16
    ```
## Test TensorRT model
  + In LightTrack repo, modify engine path in `lib/models/lighttrack_speed_trt.py`, then run:
    ```python
    python3 tracking/test_lighttrack_trt_uav.py --arch LightTrackM_Subnet --dataset UAV123 --stride 16 --even 0 --path_name back_04502514044521042540+cls_211000022+reg_100000111_ops_32
    ```
## Evaluate performance
 + Put the results folder under pysot, evaluation please refer to [pysot](https://github.com/STVIR/pysot).
