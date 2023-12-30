# Punica integreted with SmoothQuant
## Usage Instruction 
### Init Python Env
Baselines are evaluated on CUDA 12.1, check [README.md](../../kernels/baselines/README.md) to setup env.
```
cd /PATH_TO_ATOM/e2e/punica-int8
conda create -n e2e-int8 python=3.10
conda activate e2e-int8
pip install -r requirements.txt
env TORCH_CUDA_ARCH_LIST="8.9" python setup.py develop
```
### Install SmoothQuant kernel operators
Install [SmoothQuant](https://github.com/mit-han-lab/smoothquant/tree/main) kernel operator. If encountered compile error, please change compile flag in `setup.py` into CXX17.
```
cd /PATH_TO_ATOM/kernels/3rdparty/torch-int
conda install -c anaconda gxx_linux-64=9
source environment.sh
python setup.py install
```
### E2E Throughput Test
```
cd /PATH_TO_ATOM/e2e/punica-int8
python -m benchmarks.bench_textgen --batch-size 32 --num-batches 20
```
Sample Output:
```
num_requests: 640
batch_size: 32
encode_latency: 119.320ms ± 334.211ms per request; 13.031ms ± 54.633ms per token
decode_latency: 26.765ms ± 2.807ms per token
total prompt tokens: 15004
total new tokens: 636994
duration: 548.071s
throughput ((prompt+new)/duration): 1189.624 token/s
```