# openpose-engine
OpenPose network tensorrt optimizer

It is a personal TensorRT project, so it does not related to NV's official projects.
This project aims to optimize [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) models. This project has Custom PReLU Plugin Layer for *pose/body_25* model. While I used *__half* data type, I didn't fully profiled this layer, so I don't have much to telling about this at this time.

# How to use

## 1. Clone & Mount this with TensorRT NGC docker image
```bash
$ git clone https://github.com/haanjack/openpose-engine
# nvidia-docker 2.0
$ docker run -d --runtime=nvidia --name=tensorrt -v $(pwd)/openpose-engine:/workspace \
    nvcr.io/nvidia/tensorrt:18.09-py3
```

## 2. Download models
```bash
$ ./models/getModels.sh
```

## 3. Build & Run
```bash
$ docker exec -ti -e VERBOSE=1 tensorrt make
$ docker exec -ti -e tensorrt bin/openpose --output=net_output \
    --deploy=models/pose/body_25/pose_deploy.prototxt --model=models/pose/body_25/pose_iter_584000.caffemodel \
    --device=1 --batch=4 --fp16
```

# Sample results using 1x Tesla V100-DGX-Station
| model | batch size | fp16 |
|:--- | --- | --- |
| models/pose/body_25 | 4 | O |
```bash
output: net_output
device: 1
batch: 4
deploy: models/pose/body_25/pose_deploy.prototxt
model: models/pose/body_25/pose_iter_584000.caffemodel
fp16
Building and running a GPU inference engine for OpenPose, N=4...
Input "image": 3x640x480
Output "net_output": 78x80x60
Run inference...
name=image, bindingIndex=0, buffers.size()=2
name=net_output, bindingIndex=1, buffers.size()=2
Average over 10 runs is 25.9082 ms (host walltime is 26.0956 ms, 99% percentile time is 25.9656).
Average over 10 runs is 25.8934 ms (host walltime is 26.0002 ms, 99% percentile time is 26.0024).
Average over 10 runs is 25.8963 ms (host walltime is 25.9705 ms, 99% percentile time is 25.986).
Average over 10 runs is 25.9304 ms (host walltime is 26.1024 ms, 99% percentile time is 26.0352).
Average over 10 runs is 25.9703 ms (host walltime is 26.046 ms, 99% percentile time is 26.0516).
Average over 10 runs is 25.9631 ms (host walltime is 26.0303 ms, 99% percentile time is 26.111).
Average over 10 runs is 25.9284 ms (host walltime is 26.0057 ms, 99% percentile time is 26.0096).
Average over 10 runs is 25.9412 ms (host walltime is 26.0246 ms, 99% percentile time is 25.984).
Average over 10 runs is 25.9781 ms (host walltime is 26.0588 ms, 99% percentile time is 26.0966).
Average over 10 runs is 25.9397 ms (host walltime is 26.0214 ms, 99% percentile time is 26.0157).
Done.
```

| model | batch size | fp16 |
|:--- | --- | --- |
| models/pose/body_25 | 1 | O |
```
output: net_output
device: 1
batch: 1
deploy: models/pose/body_25/pose_deploy.prototxt
model: models/pose/body_25/pose_iter_584000.caffemodel
fp16
Building and running a GPU inference engine for OpenPose, N=1...
Input "image": 3x640x480
Output "net_output": 78x80x60
Run inference...
name=image, bindingIndex=0, buffers.size()=2
name=net_output, bindingIndex=1, buffers.size()=2
Average over 10 runs is 10.6626 ms (host walltime is 10.723 ms, 99% percentile time is 10.6998).
Average over 10 runs is 10.654 ms (host walltime is 10.7081 ms, 99% percentile time is 10.6762).
Average over 10 runs is 10.6534 ms (host walltime is 10.7549 ms, 99% percentile time is 10.6988).
Average over 10 runs is 10.6386 ms (host walltime is 10.6966 ms, 99% percentile time is 10.6619).
Average over 10 runs is 10.6651 ms (host walltime is 10.7307 ms, 99% percentile time is 10.6875).
Average over 10 runs is 10.7066 ms (host walltime is 10.7797 ms, 99% percentile time is 10.7459).
Average over 10 runs is 10.667 ms (host walltime is 10.7657 ms, 99% percentile time is 10.6967).
Average over 10 runs is 10.7005 ms (host walltime is 10.7898 ms, 99% percentile time is 10.7418).
Average over 10 runs is 10.7307 ms (host walltime is 10.8071 ms, 99% percentile time is 10.7704).
Average over 10 runs is 10.7342 ms (host walltime is 10.809 ms, 99% percentile time is 10.7725).
Done.
```

# Todo
* [ ] Integration with gie sample for general use
* [ ] TensorRT Plan file I/O
* [ ] Integration with OpenPose Application
