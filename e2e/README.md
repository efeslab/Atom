# End-to-end efficiency evaluation of Atom in Serving Context
## Experiment Setup
[Punica](https://github.com/punica-ai/punica) is a serving framework dedicated for serving multiple fine-tuned LoRA models. Atom uses its backbone framework without LoRA part to demonstrate its' efficiency in serving context. This codebase is modified from previous version of Punica.

To evaluate the real production scenario, we collect real-world LLM chat logs from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json), from which we sample prefill prompts and decoding length to synthesize user requests. We adopt continous batching following [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) and manually set the corresponding batch size.

The backbone inference workflow of Punica is based on PyTorch [Huggingface Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). We only subsitute corresponding kernels for each tested methods. For W8A8 evaluation, we apply [SmoothQuant](https://github.com/mit-han-lab/smoothquant) and also replace the attention kernel with FP8 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) implementation. For W4A16 evaluation, we utilize kernels from [AWQ](https://github.com/mit-han-lab/llm-awq/tree/main/awq) and replace all linear layers with AWQ quantized versions.

Note that current codebase is for efficiency evaluation. We use random weights and hack therefore no meaningful output.
## Usage Instruction
Check README.md in each folder for detailed instructions.
## Results
![e2e](../figures/atom_e2e_eval.png)