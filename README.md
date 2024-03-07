# Atom: Low-bit Quantization for Efficient and Accurate LLM Serving


## Abstract

## Installation
1. Cloning this remo and 3rdparty library
```
git clone --recurse-submodules https://github.com/efeslab/Atom4LowRank
cd Atom
```
2. Prepare environment
```
cd model
conda create -n atom python=3.10
conda activate atom
pip install -r requirements.txt
```

## Usage
### Accuracy Evaluation
Before running this command, please generate the ASVD model following the steps provided in `3rdparty/ASVD4LLM`

```
python model/main.py ./Llama-2-7b-hf-asvd80 wikitext2 \
    --wbits 8 --abits 8 --a_sym --w_sym \
    --kv_cache \
    --eval_ppl 
```

