# Punica integreted with SmoothQuant
## Usage Instruction 
### Init Python Env
Baselines are evaluated on CUDA 12.1.
```
conda create -n e2e-int8 python=3.10
conda activate e2e-int8
pip install -r requirements.txt
```
### Install Punica and SmoothQuant kernel operators
Check [SmoothQuant](https://github.com/mit-han-lab/smoothquant/tree/main) to install SmoothQuant kernel operators (`torch-int`).
```
env TORCH_CUDA_ARCH_LIST="8.9" python setup.py develop
```
### E2E Throughput Test
```
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