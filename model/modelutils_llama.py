import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from qLinearLayer import find_qlinear_layers
from qLlamaLayer import QLlamaDecoderLayer
from gptq import GPTQ, Quantizer_GPTQ
from functools import partial

from quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper

def reorder_model_llama(model, device, args, reorder_index):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"


    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]
        
        nameTemplate = 'layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        m.mlp.gate_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')]
        )
        m.mlp.up_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')]
        )
        m.mlp.down_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            out_reorder_index=None
        )
        # K has outlier should be kept.
        # Not reorder due to the RoPE embedding.
        m.self_attn.q_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            # out_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'output')]
            out_reorder_index=None
        )
        m.self_attn.k_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            # out_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'output')]
            out_reorder_index=None
        )
        
        m.self_attn.v_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.o_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            out_reorder_index=None
        )
        m.input_layernorm.register_buffer('reorder_index', 
            reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )
        m.post_attention_layernorm.register_buffer('reorder_index',
            reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')]
        )
        m.self_attn.register_buffer('reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')])

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def add_act_quant_wrapper_llama(model, device, args, scales):
    model.config.use_cache = False
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)

        nameTemplate = 'layers.{}.{}.{}'
        m.self_attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args),
            scales[nameTemplate.format(i, 'self_attn', 'o_proj')]
        )
        m.self_attn.v_quant.configure(
            partial(quantize_attn_v_wrapper, args=args),
            None
        )
        m.self_attn.k_quant.configure(
            partial(quantize_attn_k_wrapper, args=args),
            None
        )

        m.mlp.act_quant.configure(
            partial(quantize_activation_wrapper, args=args),
            scales[nameTemplate.format(i, 'mlp', 'down_proj')]
        )
        m.input_layernorm.act_quant.configure(
            partial(quantize_activation_wrapper, args=args),
            scales[nameTemplate.format(i, 'self_attn', 'k_proj')]
        )
        m.post_attention_layernorm.act_quant.configure(
            partial(quantize_activation_wrapper, args=args),
            scales[nameTemplate.format(i, 'mlp', 'gate_proj')]
        )
        
        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model_llama(model, device, args):
    model.config.use_cache = False
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)
        m.mlp.gate_proj.quant()
        m.mlp.up_proj.quant()
        m.mlp.down_proj.quant()
        m.self_attn.q_proj.quant()
        m.self_attn.k_proj.quant()
        m.self_attn.v_proj.quant()
        m.self_attn.o_proj.quant()

        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model_gptq_llama(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        layer = m.to(device)

        block_layers = find_qlinear_layers(layer)

        sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(
                    subset[name], n_out=args.keeper, keeper_precision=args.keeper_precision
                )
                gptq[name].quantizer = Quantizer_GPTQ()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.w_sym, mse=False, 
                    channel_group=args.weight_channel_group,
                    clip_ratio=args.w_clip_ratio
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            
            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.weight_group_size
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()

            del gptq

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer, m
        torch.cuda.empty_cache()
        gc.collect()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model