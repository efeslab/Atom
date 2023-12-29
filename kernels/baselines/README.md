# Atom: Baseline Kernel Evaluations
## Environment Setup
We evaluate baselines in CUDA 12.1 to maxmize their performance. Follow the instructions to setup the container.
```
docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
docker run -it --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 /bin/bash
```
Make sure you install wget, git, conda and cmake (>= 3.24). We use [NVBench](https://github.com/NVIDIA/nvbench.git) to evaluate the kernel performance and we need [libTorch](https://pytorch.org/) with `_GLIBCXX_USE_CXX11_ABI = 1` to make baselines compatible with NVBench. Follow the instructions below to setup the environment.
```
git clone --recurse-submodules https://github.com/efeslab/Atom

wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
mv libtorch /PATH_TO_ATOM/kernels/3rdparty/
```
Install Python dev to include `Python.h` for torch extension.
```
apt-get install python3-dev
```
Use the following instructions or scripts `build.sh` to build the baseline benchmark.
```
cd /PATH_TO_ATOM/kernels/baselines
mkdir build
cd build
# Fill in your libtorch path
cmake .. -DCMAKE_PREFIX_PATH=/PATH_TO_ATOM/kernels/3rdparty/libtorch
make -j
```
## Result
8-bit Weight-activation Quantization (SmoothQuant) and 4-bit Weight-only Quantization (AWQ) are evaluated in CUDA 12.1 to maximize their performance. Note that `Elem/s` denotes the computation throughput (Flops/s).

W8A8 Evaluation `./bench_torch_int`:
![SmoothQuant](../../figures/bench_torch_int.png)

W4A16 Evaluation `./bench_awq`:
![AWQ](../../figures/bench_awq.png)

We also use PyTorch Extension to evaluate the performance of PyTorch API Kernel. Baselines are installed according to their official codebases. Please refer to this [notebook](./python-api.ipynb) to check the results. Below is a sample figure:
<div align=center>
    <img src="../../figures/python-api.png" width="50%" height="50%">
</div>