# SiamTracker-with-TensorRT
The modify PyTorch version of Siam-trackers which are speed-up by TensorRT or ONNX. **[Updating...]**
</br> Examples demonstrating how to optimize PyTorch models with ONNX or TensorRT and do inference on NVIDIA Jetson platforms. 
   + All reported speeds contain pre-process process.

| Dataset | Tracker | Platform | Origin Success(%)/Speed(fps) | TensorRT-FP32 Success(%)/Speed(fps) | TensorRT-FP16 Success(%)/Speed(fps) | ONNX Success(%)/Speed(fps) |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| UAV123 | LightTrack | Jetson AGX Xavier | 62.6/15~ | 61.7/38~ | 61.3/42~ | 61.6/36~ |

## Installation
### Prerequisites
The code in this repository was tested on Jetson AGX Xavier DevKits. In order to go futher, 
first make sure you have the proper version of image (JetPack) installed on the target Jetson system. 
If not, you can refer to this: [Set up Jetson Xavier family.](https://blog.csdn.net/xiao_zhang99/article/details/121704925?spm=1001.2014.3001.5501)
  + More specifically, the target Jetson system must have ONNX and TensorRT libraries installed.
