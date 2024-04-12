import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from qLinearLayer import find_qlinear_layers
from qMixtralLayer import QMixtralDecoderLayer
from gptq import GPTQ, Quantizer_GPTQ
from functools import partial

from quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper

def reorder_model_mixtral(model, device, args, reorder_index):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"


    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], MixtralDecoderLayer):
            m = QMixtralDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QMixtralDecoderLayer):
            m = layers[i]

        # reordering for the attention
        nameTemplate = 'layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        m.input_layernorm.register_buffer('reorder_index', 
            reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )

        # K has outlier should be kept.
        # Not reorder due to the RoPE embedding.
        m.self_attn.q_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.k_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        
        m.self_attn.v_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.o_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            out_reorder_index=None
        )

        m.self_attn.register_buffer('reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')])
        
        # reordering for the MoE
        nameTemplate_moe = 'layers.{}.{}.{}.{}.{}.{}' # Something like layers.10.block_sparse_moe.experts.1.w1

        # pick expert.0.w1's order and reorder all related modules
        m.block_sparse_moe.gate.reorder(
            in_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w1', 'input')],
            out_reorder_index=None
        )

        num_experts = m.block_sparse_moe.num_experts
        for j in range(num_experts):
            m.block_sparse_moe.experts[j].w1.reorder(
                in_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w1', 'input')],
                out_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w2', 'input')]
            )
            m.block_sparse_moe.experts[j].w3.reorder(
                in_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w1', 'input')],
                out_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w2', 'input')]
            )
            m.block_sparse_moe.experts[j].w2.reorder(
                in_reorder_index=reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w2', 'input')],
                out_reorder_index=None
            )

        m.post_attention_layernorm.register_buffer('reorder_index',
            reorder_index[nameTemplate_moe.format(i, 'block_sparse_moe', 'experts', 0, 'w1', 'input')],
        )

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model


def add_act_quant_wrapper_mixtral(model, device, args, scales):
    model.config.use_cache = False
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        if isinstance(layers[i], MixtralDecoderLayer):
            m = QMixtralDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QMixtralDecoderLayer):
            m = layers[i]
        else:
            continue

        m = m.to(device)

        m.self_attn.act_quant = partial(quantize_activation_wrapper, args=args)
        m.self_attn.v_quant = partial(quantize_attn_v_wrapper, args=args)
        m.self_attn.k_quant = partial(quantize_attn_k_wrapper, args=args)

        for expert in m.block_sparse_moe.experts:
            expert.act_quant = partial(quantize_activation_wrapper, args=args)

        m.act_quant = partial(quantize_activation_wrapper, args=args)
        m.block_sparse_moe.act_quant = partial(quantize_activation_wrapper, args=args)
        
        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model_mixtral(model, device, args):
    model.config.use_cache = False
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        if isinstance(layers[i], MixtralDecoderLayer):
            m = QMixtralDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QMixtralDecoderLayer):
            m = layers[i]
        else:
            continue

        m = m.to(device)
        for expert in m.block_sparse_moe.experts:
            expert.quant()

        m.self_attn.q_proj.quant()
        m.self_attn.k_proj.quant()
        m.self_attn.v_proj.quant()
        m.self_attn.o_proj.quant()

        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model