# openpose-engine
OpenPose network tensorrt optimizer


# How to use

## 1. Clone & Mount this with TensorRT NGC docker image
```bash
$ git clone https://github.com/haanjack/openpose-engine
# nvidia-docker 2.0
$ docker run -d --runtime=nvidia --name=tensorrt -v $(pwd)/openpose-engine:/workspace \
    nvcr.io/nvidia/tensorrt:18.09-py3
```

## 2. Build & Run
```bash
$ docker exec -ti -e CUDA_VISIBLE_DEVICES=0 -e VERBOSE=1 tensorrt make
$ docker exec -ti -e CUDA_VISIBLE_DEVICES=0 tensorrt bin/openpose
```

** Result using 1 Tesla V100-DGX-Station **
```bash
Building and running a GPU inference engine for OpenPose, N=4...
models/pose/body_25/pose_deploy.prototxt
use fp16..
Bindings after deserializing:
Binding 0 (image): Input.
Binding 1 (net_output): Output.
conv1_1 + relu1_1 input reformatter 0    0.054ms
conv1_1 + relu1_1                        0.539ms
conv1_2 + relu1_2                        1.090ms
pool1_stage1                             0.264ms
conv2_1 + relu2_1                        0.537ms
conv2_2 + relu2_2                        0.934ms
pool2_stage1                             0.137ms
conv3_1 + relu3_1                        0.495ms
conv3_2 + relu3_2                        0.910ms
conv3_3 + relu3_3                        0.910ms
conv3_4 + relu3_4                        0.909ms
pool3_stage1                             0.071ms
conv4_1 + relu4_1                        0.459ms
conv4_2                                  0.883ms
prelu4_2 input reformatter 0             0.054ms
prelu4_2                                 0.102ms
conv4_3_CPM input reformatter 0          0.057ms
conv4_3_CPM                              0.448ms
prelu4_3_CPM input reformatter 0         0.029ms
prelu4_3_CPM                             0.054ms
conv4_4_CPM input reformatter 0          0.032ms
conv4_4_CPM                              0.119ms
prelu4_4_CPM input reformatter 0         0.015ms
prelu4_4_CPM                             0.028ms
Mconv1_stage0_L2_0 input reformatter 0   0.019ms
Mconv1_stage0_L2_0                       0.064ms
Mprelu1_stage0_L2_0 input reformatter 0  0.012ms
Mprelu1_stage0_L2_0                      0.016ms
Mconv1_stage0_L2_1 input reformatter 0   0.014ms
Mconv1_stage0_L2_1                       0.051ms
Mprelu1_stage0_L2_1 input reformatter 0  0.012ms
Mprelu1_stage0_L2_1                      0.024ms
Mconv1_stage0_L2_2 input reformatter 0   0.014ms
Mconv1_stage0_L2_2                       0.051ms
Mprelu1_stage0_L2_2 input reformatter 0  0.012ms
Mprelu1_stage0_L2_2                      0.024ms
Mconv1_stage0_L2_0 copy                  0.017ms
Mconv1_stage0_L2_1 copy                  0.016ms
Mconv1_stage0_L2_2 copy                  0.016ms
Mconv2_stage0_L2_0                       0.131ms
Mprelu2_stage0_L2_0 input reformatter 0  0.012ms
Mprelu2_stage0_L2_0                      0.024ms
Mconv2_stage0_L2_1 input reformatter 0   0.014ms
Mconv2_stage0_L2_1                       0.051ms
Mprelu2_stage0_L2_1 input reformatter 0  0.012ms
Mprelu2_stage0_L2_1                      0.023ms
Mconv2_stage0_L2_2 input reformatter 0   0.014ms
Mconv2_stage0_L2_2                       0.051ms
Mprelu2_stage0_L2_2 input reformatter 0  0.012ms
Mprelu2_stage0_L2_2                      0.016ms
Mconv2_stage0_L2_0 copy                  0.017ms
Mconv2_stage0_L2_1 copy                  0.016ms
Mconv2_stage0_L2_2 copy                  0.016ms
Mconv3_stage0_L2_0                       0.131ms
Mprelu3_stage0_L2_0 input reformatter 0  0.012ms
Mprelu3_stage0_L2_0                      0.024ms
Mconv3_stage0_L2_1 input reformatter 0   0.014ms
Mconv3_stage0_L2_1                       0.050ms
Mprelu3_stage0_L2_1 input reformatter 0  0.012ms
Mprelu3_stage0_L2_1                      0.024ms
Mconv3_stage0_L2_2 input reformatter 0   0.014ms
Mconv3_stage0_L2_2                       0.051ms
Mprelu3_stage0_L2_2 input reformatter 0  0.012ms
Mprelu3_stage0_L2_2                      0.024ms
Mconv3_stage0_L2_0 copy                  0.017ms
Mconv3_stage0_L2_1 copy                  0.016ms
Mconv3_stage0_L2_2 copy                  0.016ms
Mconv4_stage0_L2_0                       0.131ms
Mprelu4_stage0_L2_0 input reformatter 0  0.012ms
Mprelu4_stage0_L2_0                      0.024ms
Mconv4_stage0_L2_1 input reformatter 0   0.014ms
Mconv4_stage0_L2_1                       0.051ms
Mprelu4_stage0_L2_1 input reformatter 0  0.012ms
Mprelu4_stage0_L2_1                      0.024ms
Mconv4_stage0_L2_2 input reformatter 0   0.014ms
Mconv4_stage0_L2_2                       0.050ms
Mprelu4_stage0_L2_2 input reformatter 0  0.012ms
Mprelu4_stage0_L2_2                      0.021ms
Mconv4_stage0_L2_0 copy                  0.016ms
Mconv4_stage0_L2_1 copy                  0.016ms
Mconv4_stage0_L2_2 copy                  0.016ms
Mconv5_stage0_L2_0                       0.131ms
Mprelu5_stage0_L2_0 input reformatter 0  0.012ms
Mprelu5_stage0_L2_0                      0.024ms
Mconv5_stage0_L2_1 input reformatter 0   0.014ms
Mconv5_stage0_L2_1                       0.051ms
Mprelu5_stage0_L2_1 input reformatter 0  0.012ms
Mprelu5_stage0_L2_1                      0.022ms
Mconv5_stage0_L2_2 input reformatter 0   0.014ms
Mconv5_stage0_L2_2                       0.051ms
Mprelu5_stage0_L2_2 input reformatter 0  0.012ms
Mprelu5_stage0_L2_2                      0.017ms
Mconv5_stage0_L2_0 copy                  0.017ms
Mconv5_stage0_L2_1 copy                  0.016ms
Mconv5_stage0_L2_2 copy                  0.016ms
Mconv6_stage0_L2                         0.054ms
Mprelu6_stage0_L2 input reformatter 0    0.029ms
Mprelu6_stage0_L2                        0.053ms
Mconv7_stage0_L2 input reformatter 0     0.032ms
Mconv7_stage0_L2                         0.026ms
conv4_4_CPM copy                         0.106ms
Mconv1_stage1_L2_0                       0.095ms
Mprelu1_stage1_L2_0 input reformatter 0  0.015ms
Mprelu1_stage1_L2_0                      0.020ms
Mconv1_stage1_L2_1 input reformatter 0   0.018ms
Mconv1_stage1_L2_1                       0.066ms
Mprelu1_stage1_L2_1 input reformatter 0  0.015ms
Mprelu1_stage1_L2_1                      0.030ms
Mconv1_stage1_L2_2 input reformatter 0   0.019ms
Mconv1_stage1_L2_2                       0.066ms
Mprelu1_stage1_L2_2 input reformatter 0  0.015ms
Mprelu1_stage1_L2_2                      0.028ms
Mconv1_stage1_L2_0 copy                  0.019ms
Mconv1_stage1_L2_1 copy                  0.019ms
Mconv1_stage1_L2_2 copy                  0.019ms
Mconv2_stage1_L2_0                       0.175ms
Mprelu2_stage1_L2_0 input reformatter 0  0.015ms
Mprelu2_stage1_L2_0                      0.030ms
Mconv2_stage1_L2_1 input reformatter 0   0.019ms
Mconv2_stage1_L2_1                       0.066ms
Mprelu2_stage1_L2_1 input reformatter 0  0.015ms
Mprelu2_stage1_L2_1                      0.030ms
Mconv2_stage1_L2_2 input reformatter 0   0.018ms
Mconv2_stage1_L2_2                       0.066ms
Mprelu2_stage1_L2_2 input reformatter 0  0.015ms
Mprelu2_stage1_L2_2                      0.022ms
Mconv2_stage1_L2_0 copy                  0.019ms
Mconv2_stage1_L2_1 copy                  0.019ms
Mconv2_stage1_L2_2 copy                  0.019ms
Mconv3_stage1_L2_0                       0.175ms
Mprelu3_stage1_L2_0 input reformatter 0  0.015ms
Mprelu3_stage1_L2_0                      0.030ms
Mconv3_stage1_L2_1 input reformatter 0   0.018ms
Mconv3_stage1_L2_1                       0.066ms
Mprelu3_stage1_L2_1 input reformatter 0  0.015ms
Mprelu3_stage1_L2_1                      0.030ms
Mconv3_stage1_L2_2 input reformatter 0   0.019ms
Mconv3_stage1_L2_2                       0.066ms
Mprelu3_stage1_L2_2 input reformatter 0  0.015ms
Mprelu3_stage1_L2_2                      0.030ms
Mconv3_stage1_L2_0 copy                  0.019ms
Mconv3_stage1_L2_1 copy                  0.019ms
Mconv3_stage1_L2_2 copy                  0.019ms
Mconv4_stage1_L2_0                       0.175ms
Mprelu4_stage1_L2_0 input reformatter 0  0.015ms
Mprelu4_stage1_L2_0                      0.029ms
Mconv4_stage1_L2_1 input reformatter 0   0.019ms
Mconv4_stage1_L2_1                       0.066ms
Mprelu4_stage1_L2_1 input reformatter 0  0.015ms
Mprelu4_stage1_L2_1                      0.028ms
Mconv4_stage1_L2_2 input reformatter 0   0.018ms
Mconv4_stage1_L2_2                       0.066ms
Mprelu4_stage1_L2_2 input reformatter 0  0.015ms
Mprelu4_stage1_L2_2                      0.029ms
Mconv4_stage1_L2_0 copy                  0.019ms
Mconv4_stage1_L2_1 copy                  0.019ms
Mconv4_stage1_L2_2 copy                  0.018ms
Mconv5_stage1_L2_0                       0.175ms
Mprelu5_stage1_L2_0 input reformatter 0  0.015ms
Mprelu5_stage1_L2_0                      0.030ms
Mconv5_stage1_L2_1 input reformatter 0   0.019ms
Mconv5_stage1_L2_1                       0.066ms
Mprelu5_stage1_L2_1 input reformatter 0  0.015ms
Mprelu5_stage1_L2_1                      0.030ms
Mconv5_stage1_L2_2 input reformatter 0   0.018ms
Mconv5_stage1_L2_2                       0.066ms
Mprelu5_stage1_L2_2 input reformatter 0  0.015ms
Mprelu5_stage1_L2_2                      0.026ms
Mconv5_stage1_L2_0 copy                  0.019ms
Mconv5_stage1_L2_1 copy                  0.019ms
Mconv5_stage1_L2_2 copy                  0.018ms
Mconv6_stage1_L2                         0.131ms
Mprelu6_stage1_L2 input reformatter 0    0.054ms
Mprelu6_stage1_L2                        0.101ms
Mconv7_stage1_L2 input reformatter 0     0.057ms
Mconv7_stage1_L2                         0.040ms
Mconv1_stage2_L2_0                       0.095ms
Mprelu1_stage2_L2_0 input reformatter 0  0.015ms
Mprelu1_stage2_L2_0                      0.020ms
Mconv1_stage2_L2_1 input reformatter 0   0.019ms
Mconv1_stage2_L2_1                       0.066ms
Mprelu1_stage2_L2_1 input reformatter 0  0.015ms
Mprelu1_stage2_L2_1                      0.030ms
Mconv1_stage2_L2_2 input reformatter 0   0.019ms
Mconv1_stage2_L2_2                       0.066ms
Mprelu1_stage2_L2_2 input reformatter 0  0.015ms
Mprelu1_stage2_L2_2                      0.030ms
Mconv1_stage2_L2_0 copy                  0.019ms
Mconv1_stage2_L2_1 copy                  0.019ms
Mconv1_stage2_L2_2 copy                  0.019ms
Mconv2_stage2_L2_0                       0.175ms
Mprelu2_stage2_L2_0 input reformatter 0  0.015ms
Mprelu2_stage2_L2_0                      0.030ms
Mconv2_stage2_L2_1 input reformatter 0   0.019ms
Mconv2_stage2_L2_1                       0.066ms
Mprelu2_stage2_L2_1 input reformatter 0  0.015ms
Mprelu2_stage2_L2_1                      0.030ms
Mconv2_stage2_L2_2 input reformatter 0   0.019ms
Mconv2_stage2_L2_2                       0.066ms
Mprelu2_stage2_L2_2 input reformatter 0  0.015ms
Mprelu2_stage2_L2_2                      0.025ms
Mconv2_stage2_L2_0 copy                  0.019ms
Mconv2_stage2_L2_1 copy                  0.019ms
Mconv2_stage2_L2_2 copy                  0.019ms
Mconv3_stage2_L2_0                       0.175ms
Mprelu3_stage2_L2_0 input reformatter 0  0.015ms
Mprelu3_stage2_L2_0                      0.027ms
Mconv3_stage2_L2_1 input reformatter 0   0.019ms
Mconv3_stage2_L2_1                       0.066ms
Mprelu3_stage2_L2_1 input reformatter 0  0.015ms
Mprelu3_stage2_L2_1                      0.029ms
Mconv3_stage2_L2_2 input reformatter 0   0.018ms
Mconv3_stage2_L2_2                       0.066ms
Mprelu3_stage2_L2_2 input reformatter 0  0.015ms
Mprelu3_stage2_L2_2                      0.029ms
Mconv3_stage2_L2_0 copy                  0.019ms
Mconv3_stage2_L2_1 copy                  0.019ms
Mconv3_stage2_L2_2 copy                  0.018ms
Mconv4_stage2_L2_0                       0.175ms
Mprelu4_stage2_L2_0 input reformatter 0  0.015ms
Mprelu4_stage2_L2_0                      0.028ms
Mconv4_stage2_L2_1 input reformatter 0   0.019ms
Mconv4_stage2_L2_1                       0.066ms
Mprelu4_stage2_L2_1 input reformatter 0  0.015ms
Mprelu4_stage2_L2_1                      0.028ms
Mconv4_stage2_L2_2 input reformatter 0   0.018ms
Mconv4_stage2_L2_2                       0.066ms
Mprelu4_stage2_L2_2 input reformatter 0  0.015ms
Mprelu4_stage2_L2_2                      0.028ms
Mconv4_stage2_L2_0 copy                  0.019ms
Mconv4_stage2_L2_1 copy                  0.019ms
Mconv4_stage2_L2_2 copy                  0.018ms
Mconv5_stage2_L2_0                       0.175ms
Mprelu5_stage2_L2_0 input reformatter 0  0.015ms
Mprelu5_stage2_L2_0                      0.029ms
Mconv5_stage2_L2_1 input reformatter 0   0.019ms
Mconv5_stage2_L2_1                       0.066ms
Mprelu5_stage2_L2_1 input reformatter 0  0.015ms
Mprelu5_stage2_L2_1                      0.030ms
Mconv5_stage2_L2_2 input reformatter 0   0.018ms
Mconv5_stage2_L2_2                       0.066ms
Mprelu5_stage2_L2_2 input reformatter 0  0.015ms
Mprelu5_stage2_L2_2                      0.026ms
Mconv5_stage2_L2_0 copy                  0.019ms
Mconv5_stage2_L2_1 copy                  0.019ms
Mconv5_stage2_L2_2 copy                  0.018ms
Mconv6_stage2_L2                         0.131ms
Mprelu6_stage2_L2 input reformatter 0    0.054ms
Mprelu6_stage2_L2                        0.101ms
Mconv7_stage2_L2 input reformatter 0     0.057ms
Mconv7_stage2_L2                         0.040ms
Mconv1_stage3_L2_0                       0.095ms
Mprelu1_stage3_L2_0 input reformatter 0  0.015ms
Mprelu1_stage3_L2_0                      0.021ms
Mconv1_stage3_L2_1 input reformatter 0   0.019ms
Mconv1_stage3_L2_1                       0.066ms
Mprelu1_stage3_L2_1 input reformatter 0  0.015ms
Mprelu1_stage3_L2_1                      0.030ms
Mconv1_stage3_L2_2 input reformatter 0   0.018ms
Mconv1_stage3_L2_2                       0.066ms
Mprelu1_stage3_L2_2 input reformatter 0  0.015ms
Mprelu1_stage3_L2_2                      0.029ms
Mconv1_stage3_L2_0 copy                  0.019ms
Mconv1_stage3_L2_1 copy                  0.019ms
Mconv1_stage3_L2_2 copy                  0.018ms
Mconv2_stage3_L2_0                       0.175ms
Mprelu2_stage3_L2_0 input reformatter 0  0.015ms
Mprelu2_stage3_L2_0                      0.030ms
Mconv2_stage3_L2_1 input reformatter 0   0.019ms
Mconv2_stage3_L2_1                       0.066ms
Mprelu2_stage3_L2_1 input reformatter 0  0.015ms
Mprelu2_stage3_L2_1                      0.030ms
Mconv2_stage3_L2_2 input reformatter 0   0.019ms
Mconv2_stage3_L2_2                       0.066ms
Mprelu2_stage3_L2_2 input reformatter 0  0.015ms
Mprelu2_stage3_L2_2                      0.026ms
Mconv2_stage3_L2_0 copy                  0.019ms
Mconv2_stage3_L2_1 copy                  0.019ms
Mconv2_stage3_L2_2 copy                  0.018ms
Mconv3_stage3_L2_0                       0.174ms
Mprelu3_stage3_L2_0 input reformatter 0  0.015ms
Mprelu3_stage3_L2_0                      0.030ms
Mconv3_stage3_L2_1 input reformatter 0   0.018ms
Mconv3_stage3_L2_1                       0.066ms
Mprelu3_stage3_L2_1 input reformatter 0  0.015ms
Mprelu3_stage3_L2_1                      0.030ms
Mconv3_stage3_L2_2 input reformatter 0   0.019ms
Mconv3_stage3_L2_2                       0.066ms
Mprelu3_stage3_L2_2 input reformatter 0  0.015ms
Mprelu3_stage3_L2_2                      0.030ms
Mconv3_stage3_L2_0 copy                  0.019ms
Mconv3_stage3_L2_1 copy                  0.019ms
Mconv3_stage3_L2_2 copy                  0.019ms
Mconv4_stage3_L2_0                       0.174ms
Mprelu4_stage3_L2_0 input reformatter 0  0.015ms
Mprelu4_stage3_L2_0                      0.030ms
Mconv4_stage3_L2_1 input reformatter 0   0.019ms
Mconv4_stage3_L2_1                       0.066ms
Mprelu4_stage3_L2_1 input reformatter 0  0.015ms
Mprelu4_stage3_L2_1                      0.027ms
Mconv4_stage3_L2_2 input reformatter 0   0.018ms
Mconv4_stage3_L2_2                       0.066ms
Mprelu4_stage3_L2_2 input reformatter 0  0.015ms
Mprelu4_stage3_L2_2                      0.029ms
Mconv4_stage3_L2_0 copy                  0.019ms
Mconv4_stage3_L2_1 copy                  0.019ms
Mconv4_stage3_L2_2 copy                  0.019ms
Mconv5_stage3_L2_0                       0.175ms
Mprelu5_stage3_L2_0 input reformatter 0  0.015ms
Mprelu5_stage3_L2_0                      0.030ms
Mconv5_stage3_L2_1 input reformatter 0   0.019ms
Mconv5_stage3_L2_1                       0.066ms
Mprelu5_stage3_L2_1 input reformatter 0  0.015ms
Mprelu5_stage3_L2_1                      0.029ms
Mconv5_stage3_L2_2 input reformatter 0   0.018ms
Mconv5_stage3_L2_2                       0.066ms
Mprelu5_stage3_L2_2 input reformatter 0  0.015ms
Mprelu5_stage3_L2_2                      0.027ms
Mconv5_stage3_L2_0 copy                  0.018ms
Mconv5_stage3_L2_1 copy                  0.019ms
Mconv5_stage3_L2_2 copy                  0.020ms
Mconv6_stage3_L2                         0.131ms
Mprelu6_stage3_L2 input reformatter 0    0.054ms
Mprelu6_stage3_L2                        0.100ms
Mconv7_stage3_L2 input reformatter 0     0.057ms
Mconv7_stage3_L2                         0.040ms
Mconv1_stage0_L1_0                       0.095ms
Mprelu1_stage0_L1_0 input reformatter 0  0.012ms
Mprelu1_stage0_L1_0                      0.021ms
Mconv1_stage0_L1_1 input reformatter 0   0.014ms
Mconv1_stage0_L1_1                       0.051ms
Mprelu1_stage0_L1_1 input reformatter 0  0.012ms
Mprelu1_stage0_L1_1                      0.024ms
Mconv1_stage0_L1_2 input reformatter 0   0.014ms
Mconv1_stage0_L1_2                       0.051ms
Mprelu1_stage0_L1_2 input reformatter 0  0.012ms
Mprelu1_stage0_L1_2                      0.023ms
Mconv1_stage0_L1_0 copy                  0.017ms
Mconv1_stage0_L1_1 copy                  0.016ms
Mconv1_stage0_L1_2 copy                  0.016ms
Mconv2_stage0_L1_0                       0.131ms
Mprelu2_stage0_L1_0 input reformatter 0  0.012ms
Mprelu2_stage0_L1_0                      0.024ms
Mconv2_stage0_L1_1 input reformatter 0   0.014ms
Mconv2_stage0_L1_1                       0.051ms
Mprelu2_stage0_L1_1 input reformatter 0  0.012ms
Mprelu2_stage0_L1_1                      0.024ms
Mconv2_stage0_L1_2 input reformatter 0   0.014ms
Mconv2_stage0_L1_2                       0.051ms
Mprelu2_stage0_L1_2 input reformatter 0  0.012ms
Mprelu2_stage0_L1_2                      0.024ms
Mconv2_stage0_L1_0 copy                  0.017ms
Mconv2_stage0_L1_1 copy                  0.016ms
Mconv2_stage0_L1_2 copy                  0.016ms
Mconv3_stage0_L1_0                       0.131ms
Mprelu3_stage0_L1_0 input reformatter 0  0.012ms
Mprelu3_stage0_L1_0                      0.023ms
Mconv3_stage0_L1_1 input reformatter 0   0.014ms
Mconv3_stage0_L1_1                       0.051ms
Mprelu3_stage0_L1_1 input reformatter 0  0.012ms
Mprelu3_stage0_L1_1                      0.023ms
Mconv3_stage0_L1_2 input reformatter 0   0.014ms
Mconv3_stage0_L1_2                       0.051ms
Mprelu3_stage0_L1_2 input reformatter 0  0.012ms
Mprelu3_stage0_L1_2                      0.023ms
Mconv3_stage0_L1_0 copy                  0.017ms
Mconv3_stage0_L1_1 copy                  0.016ms
Mconv3_stage0_L1_2 copy                  0.016ms
Mconv4_stage0_L1_0                       0.131ms
Mprelu4_stage0_L1_0 input reformatter 0  0.012ms
Mprelu4_stage0_L1_0                      0.022ms
Mconv4_stage0_L1_1 input reformatter 0   0.014ms
Mconv4_stage0_L1_1                       0.051ms
Mprelu4_stage0_L1_1 input reformatter 0  0.012ms
Mprelu4_stage0_L1_1                      0.024ms
Mconv4_stage0_L1_2 input reformatter 0   0.014ms
Mconv4_stage0_L1_2                       0.051ms
Mprelu4_stage0_L1_2 input reformatter 0  0.012ms
Mprelu4_stage0_L1_2                      0.024ms
Mconv4_stage0_L1_0 copy                  0.017ms
Mconv4_stage0_L1_1 copy                  0.016ms
Mconv4_stage0_L1_2 copy                  0.016ms
Mconv5_stage0_L1_0                       0.131ms
Mprelu5_stage0_L1_0 input reformatter 0  0.012ms
Mprelu5_stage0_L1_0                      0.024ms
Mconv5_stage0_L1_1 input reformatter 0   0.014ms
Mconv5_stage0_L1_1                       0.051ms
Mprelu5_stage0_L1_1 input reformatter 0  0.012ms
Mprelu5_stage0_L1_1                      0.021ms
Mconv5_stage0_L1_2 input reformatter 0   0.014ms
Mconv5_stage0_L1_2                       0.051ms
Mprelu5_stage0_L1_2 input reformatter 0  0.012ms
Mprelu5_stage0_L1_2                      0.021ms
Mconv5_stage0_L1_0 copy                  0.017ms
Mconv5_stage0_L1_1 copy                  0.016ms
Mconv5_stage0_L1_2 copy                  0.016ms
Mconv6_stage0_L1                         0.054ms
Mprelu6_stage0_L1 input reformatter 0    0.029ms
Mprelu6_stage0_L1                        0.048ms
Mconv7_stage0_L1 input reformatter 0     0.032ms
Mconv7_stage0_L1                         0.026ms
Mconv7_stage3_L2 copy                    0.042ms
Mconv1_stage1_L1_0                       0.111ms
Mprelu1_stage1_L1_0 input reformatter 0  0.015ms
Mprelu1_stage1_L1_0                      0.027ms
Mconv1_stage1_L1_1 input reformatter 0   0.019ms
Mconv1_stage1_L1_1                       0.066ms
Mprelu1_stage1_L1_1 input reformatter 0  0.015ms
Mprelu1_stage1_L1_1                      0.030ms
Mconv1_stage1_L1_2 input reformatter 0   0.018ms
Mconv1_stage1_L1_2                       0.066ms
Mprelu1_stage1_L1_2 input reformatter 0  0.015ms
Mprelu1_stage1_L1_2                      0.030ms
Mconv1_stage1_L1_0 copy                  0.019ms
Mconv1_stage1_L1_1 copy                  0.019ms
Mconv1_stage1_L1_2 copy                  0.019ms
Mconv2_stage1_L1_0                       0.174ms
Mprelu2_stage1_L1_0 input reformatter 0  0.015ms
Mprelu2_stage1_L1_0                      0.030ms
Mconv2_stage1_L1_1 input reformatter 0   0.018ms
Mconv2_stage1_L1_1                       0.066ms
Mprelu2_stage1_L1_1 input reformatter 0  0.015ms
Mprelu2_stage1_L1_1                      0.030ms
Mconv2_stage1_L1_2 input reformatter 0   0.019ms
Mconv2_stage1_L1_2                       0.066ms
Mprelu2_stage1_L1_2 input reformatter 0  0.015ms
Mprelu2_stage1_L1_2                      0.030ms
Mconv2_stage1_L1_0 copy                  0.019ms
Mconv2_stage1_L1_1 copy                  0.019ms
Mconv2_stage1_L1_2 copy                  0.019ms
Mconv3_stage1_L1_0                       0.174ms
Mprelu3_stage1_L1_0 input reformatter 0  0.015ms
Mprelu3_stage1_L1_0                      0.030ms
Mconv3_stage1_L1_1 input reformatter 0   0.019ms
Mconv3_stage1_L1_1                       0.066ms
Mprelu3_stage1_L1_1 input reformatter 0  0.015ms
Mprelu3_stage1_L1_1                      0.030ms
Mconv3_stage1_L1_2 input reformatter 0   0.018ms
Mconv3_stage1_L1_2                       0.066ms
Mprelu3_stage1_L1_2 input reformatter 0  0.015ms
Mprelu3_stage1_L1_2                      0.030ms
Mconv3_stage1_L1_0 copy                  0.019ms
Mconv3_stage1_L1_1 copy                  0.019ms
Mconv3_stage1_L1_2 copy                  0.019ms
Mconv4_stage1_L1_0                       0.174ms
Mprelu4_stage1_L1_0 input reformatter 0  0.015ms
Mprelu4_stage1_L1_0                      0.029ms
Mconv4_stage1_L1_1 input reformatter 0   0.018ms
Mconv4_stage1_L1_1                       0.066ms
Mprelu4_stage1_L1_1 input reformatter 0  0.015ms
Mprelu4_stage1_L1_1                      0.030ms
Mconv4_stage1_L1_2 input reformatter 0   0.019ms
Mconv4_stage1_L1_2                       0.066ms
Mprelu4_stage1_L1_2 input reformatter 0  0.015ms
Mprelu4_stage1_L1_2                      0.030ms
Mconv4_stage1_L1_0 copy                  0.019ms
Mconv4_stage1_L1_1 copy                  0.019ms
Mconv4_stage1_L1_2 copy                  0.018ms
Mconv5_stage1_L1_0                       0.174ms
Mprelu5_stage1_L1_0 input reformatter 0  0.015ms
Mprelu5_stage1_L1_0                      0.030ms
Mconv5_stage1_L1_1 input reformatter 0   0.018ms
Mconv5_stage1_L1_1                       0.066ms
Mprelu5_stage1_L1_1 input reformatter 0  0.015ms
Mprelu5_stage1_L1_1                      0.027ms
Mconv5_stage1_L1_2 input reformatter 0   0.018ms
Mconv5_stage1_L1_2                       0.066ms
Mprelu5_stage1_L1_2 input reformatter 0  0.015ms
Mprelu5_stage1_L1_2                      0.028ms
Mconv5_stage1_L1_0 copy                  0.019ms
Mconv5_stage1_L1_1 copy                  0.019ms
Mconv5_stage1_L1_2 copy                  0.020ms
Mconv6_stage1_L1                         0.131ms
Mprelu6_stage1_L1 input reformatter 0    0.054ms
Mprelu6_stage1_L1                        0.099ms
Mconv7_stage1_L1 input reformatter 0     0.057ms
Mconv7_stage1_L1                         0.041ms
Mconv7_stage1_L1 output reformatter 0    0.007ms
Time over all layers: 25.567
```

# Todo
* [ ] Having target parameter
* [ ] Integration with OpenPose Application
