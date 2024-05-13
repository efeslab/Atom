# Punica integreted with AWQ
## Usage Instruction 
### Init Python Env
Baselines are evaluated on CUDA 12.1, check [README.md](../../kernels/baselines/README.md) to setup env.
```
cd /PATH_TO_ATOM/e2e/punica-awq
conda create -n e2e-awq python=3.10
conda activate e2e-awq
pip install -r requirements.txt
env TORCH_CUDA_ARCH_LIST="8.9" python setup.py develop
```
### Install AWQ kernel operators
[AWQ](https://github.com/mit-han-lab/llm-awq/tree/main/awq) kernels from official codebase are put into `./punica/ops/csrc/gemm` and automatically compiled during `setup.py`. Note that in `./punica/models/llama.py` we replace all linear layers with AWQ linear layers.

### E2E Throughput Test
```
cd /PATH_TO_ATOM/e2e/punica-awq
python -m benchmarks.bench_textgen --batch-size 32 --num-batches 20
```
Sample Output:
```
num_requests: 640
batch_size: 32
encode_latency: 73.656ms ± 113.295ms per request; 7.766ms ± 19.953ms per token
decode_latency: 34.905ms ± 2.910ms per token
total prompt tokens: 15004
total new tokens: 636994
duration: 712.992s
throughput ((prompt+new)/duration): 914.453 token/s
```