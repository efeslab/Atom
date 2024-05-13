# Punica with FP16 Implementation
## Usage Instruction 
### Init Python Env
Baselines are evaluated on CUDA 12.1, check [README.md](../../kernels/baselines/README.md) to setup env.
```
cd /PATH_TO_ATOM/e2e/punica-fp16
conda create -n e2e-fp16 python=3.10
conda activate e2e-fp16
pip install -r requirements.txt
env TORCH_CUDA_ARCH_LIST="8.9" python setup.py develop # As evaluated on RTX4090
```

### E2E Throughput Test
```
cd /PATH_TO_ATOM/e2e/punica-fp16
python -m benchmarks.bench_textgen --batch-size 32 --num-batches 20
```
Sample Output:
```
num_requests: 640
batch_size: 32
encode_latency: 65.761ms ± 108.769ms per request; 6.974ms ± 18.901ms per token
decode_latency: 30.121ms ± 2.027ms per token
total prompt tokens: 15004
total new tokens: 636994
duration: 622.365s
throughput ((prompt+new)/duration): 1047.613 token/s
```