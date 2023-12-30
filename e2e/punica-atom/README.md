# Punica integreted with Atom
## Usage Instruction 
### Init Python Env
Under the same CUDA 11.3 container of accuracy evaluation (Check [README.md](../../README.md) to setup env), run:
```
conda create -n e2e python=3.10
conda activate e2e
pip install -r requirements.txt
```
### Install Atom kernel operators
```
env TORCH_CUDA_ARCH_LIST="8.6" python setup.py develop
```
### E2E Throughput Test
```
python -m benchmarks.bench_textgen --batch-size 32 --num-batches 20
```
Sample Output:
```
num_requests: 640
batch_size: 32
encode_latency: 77.953ms ± 203.136ms per request; 8.441ms ± 33.332ms per token
decode_latency: 15.603ms ± 1.543ms per token
total prompt tokens: 15004
total new tokens: 636994
duration: 317.610s
throughput ((prompt+new)/duration): 2052.825 token/s
```