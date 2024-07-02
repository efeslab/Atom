# Atom: Low-bit Quantization for Efficient and Accurate LLM Serving
[[paper](https://arxiv.org/abs/2310.19102)] [[slides](./figures/atom_mlsys_slides.pdf)]  [[poster](./figures/atom_mlsys_poster.pdf)]

![overview](figures/overview_and_ppl.png)

Atom is an accurate low-bit weight-activation quantization algorithm that combines (1) mixed-precision, (2) fine-grained group quantization, (3) dynamic activation quantization, (4) KV-cache quantization, and (5) efficient CUDA kernels co-design. 

This codebase utilizes [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git) to evaluate perplexity and zero-shot accuracy. Code segments from [SmoothQuant](https://github.com/mit-han-lab/smoothquant.git), [GPTQ](https://github.com/IST-DASLab/gptq.git), and [SparseGPT](https://github.com/IST-DASLab/sparsegpt.git) are integrated to reproduce results. Our kernels are modified based on previous version of [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and tested by [NVBench](https://github.com/NVIDIA/nvbench/tree/main). Serving framework [Punica](https://github.com/punica-ai/punica) is integrated to evaluate end-to-end throughput and latency. We also use [BitsandBytes](https://github.com/TimDettmers/bitsandbytes) for new data-type evaluations (e.g., FP4). We thank the authors for their great works.

The current release features:
* Simulated quantization for accuracy evaluation.
* Perplexity and zero-shot accuracy evaluation
* Kernel benchmark & End-to-end evaluation

To do:
- [x] Release code for reproducing results.
- [x] Release code for end-to-end throughput evaluation.
- [x] Add FP4 accuracy evaluation for both weight and activation quantization.
- [x] Add support for Mixtral models.
- [ ] Optimize kernel for different GPUs.
- [ ] Full inference workflow in real production scenario.

## Abstract
The growing demand for Large Language Models (LLMs) in applications such as content generation, intelligent chatbots, and sentiment analysis poses considerable challenges for LLM service providers. To efficiently use GPU resources and boost throughput, batching multiple requests has emerged as a popular paradigm; to further speed up batching, LLM quantization techniques reduce memory consumption and increase computing capacity. However, prevalent quantization schemes (e.g., 8-bit weight-activation quantization) cannot fully leverage the capabilities of modern GPUs, such as 4-bit integer operators, resulting in sub-optimal performance.

To maximize LLMs' serving throughput, we introduce Atom, a low-bit quantization method that achieves high throughput improvements with negligible accuracy loss. Atom significantly boosts serving throughput by using low-bit operators and considerably reduces memory consumption via low-bit quantization. It attains high accuracy by applying a novel mixed-precision and fine-grained quantization process. We evaluate Atom on 4-bit weight-activation quantization setups in the serving context. Atom improves end-to-end throughput by up to 7.73× compared to the FP16 and by 2.53× compared to INT8 quantization, while maintaining the same latency target.

## Installation
1. Run in container. Mount models.
```
docker pull nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
docker run -it --gpus all -v /PATH2MODEL:/model nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 /bin/bash
```
2. Clone this repo (Make sure you install Git, and Conda)
```
git clone --recurse-submodules https://github.com/efeslab/Atom
cd Atom
```
3. Prepare environment
```
cd model
conda create -n atom python=3.10
conda activate atom
pip install -r requirements.txt
```
4. Compile kernels benchmarks (Optional): Install gcc-11 and CMake (>= 3.24)
```
apt install software-properties-common lsb-release
apt-get update

curl -s https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
apt update
apt install cmake

cd /PATH_TO_ATOM/kernels
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt install -y gcc-11 g++-11
mkdir build && cd build
cmake ..
make -j
```
## Usage
### Accuracy Evaluation
Before running this command, please download Llama model from [Hugging Face website](https://huggingface.co/models?sort=trending&search=llama) first.
We recommend downloading from [Deca-Llama](https://huggingface.co/linhvu/decapoda-research-llama-7b-hf/tree/main).

We provide several scripts to reproduce our results in the paper:

To run our W4A4 perplexity evaluation, please execute
```
bash scripts/run_atom_ppl.sh /Path/To/Llama/Model
```

To get our W4A4 zero shot accuracy on common sense tasks, please execute
```
bash scripts/run_atom_zeroshot_acc.sh /Path/To/Llama/Model
```

To run our ablation study on different quantization optimizations, please run
```
bash scripts/run_atom_ablation.sh /Path/To/Llama/Model
```


You can also customize your own quantization setup by modifying the parameters. Check [model/main.py](./model/main.py) to see the description of each parameter.
```
python model/main.py /Path/To/Llama/Model wikitext2 \
    --wbits 4 --abits 4 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 2 \
    --reorder --act_sort_metric hessian \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --keeper 128 --keeper_precision 3 --kv_cache --use_gptq \
    --eval_ppl --eval_common_sense
```
### Efficiency Evaluation
We evaluate Atom on a RTX4090 GPU. Results below are executed in [cu113](https://hub.docker.com/layers/nvidia/cuda/11.3.1-cudnn8-devel-ubuntu20.04/images/sha256-052b3b515d9653f9c6e358e5b70f8bb9d75c17a8b2039055674dfa7caa970791?context=explore) docker container. Note that current kernels are only optimized for RTX4090.

To get INT4 GEMM kernel result, please execute:
```
cd kernels/build
./bench_gemm_i4_o16
```
Check column `Elem/s` to see the computation throughput of the kernel (Flop/s).
![gemm](figures/bench_gemm.png)

Other kernel of Atom can be evaluated similarly, for e.g., `./bench_reorder`. We conduct kernel evaluation on baselines as well. Please check [baselines/README.md](./kernels/baselines/README.md) to reproduce results.

To reproduce end-to-end throughput and latency evaluation, please check [e2e/README.md](./e2e/README.md).
## Key Results
### Perplexity
We evaluate Atom's accuracy on serveral model families including Llama, Llama-2, and Mixtral, with data types of INT4 and FP4.
* WikiText2, PTB and C4 datasets on Llama family:
![perplexity](figures/atom_ppl.png)
* WikiText2 perplexity on Llama-2 and Mixtral:

  <img src="figures/atom_ppl_new.png" style="width:75%;">

### End-to-end throughput and latency
* Atom achieves up to 7.7x higher throughput with similar latency than `FP16` with a fixed GPU memory under serving scenario.
![e2e](figures/atom_e2e_eval.png)

## Reference
If you find this project is helpful to your research, please consider to cite our paper:
```
@inproceedings{MLSYS2024_5edb57c0,
 author = {Zhao, Yilong and Lin, Chien-Yu and Zhu, Kan and Ye, Zihao and Chen, Lequn and Zheng, Size and Ceze, Luis and Krishnamurthy, Arvind and Chen, Tianqi and Kasikci, Baris},
 booktitle = {Proceedings of Machine Learning and Systems},
 editor = {P. Gibbons and G. Pekhimenko and C. De Sa},
 pages = {196--209},
 title = {Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving},
 url = {https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf},
 volume = {6},
 year = {2024}
}
```
