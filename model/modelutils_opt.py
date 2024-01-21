import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from qLinearLayer import find_qlinear_layers
from qOPTLayer import QOPTDecoderLayer
from gptq import GPTQ, Quantizer_GPTQ
from functools import partial

from quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper

def reorder_model_opt(model, device, args, reorder_index):
    model.config.use_cache = False
    layers = model.model.decoder.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], OPTDecoderLayer):
            m = QOPTDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QOPTDecoderLayer):
            m = layers[i]
        
        nameTemplate_fc = 'decoder.layers.{}.{}.{}' # Something like layers.10.fc1
        nameTemplate_attn = 'decoder.layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        m.fc1.reorder(
            in_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc1', 'input')],
            out_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc2', 'input')]
        )
        m.fc2.reorder(
            in_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc2', 'input')],
            out_reorder_index=None
        )

        # K has outlier should be kept.
        # Output Not reorder due to the RoPE embedding.
        m.self_attn.q_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'q_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.k_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.v_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'v_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.out_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'out_proj', 'input')],
            out_reorder_index=None
        )

        m.self_attn_layer_norm.register_buffer('reorder_index', 
            reorder_index[nameTemplate_attn.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )
        if m.do_layer_norm_before:
            m.final_layer_norm.register_buffer('reorder_index',
                reorder_index[nameTemplate_fc.format(i, 'fc1', 'input')]
            )
        m.self_attn.register_buffer(
            'out_reorder_index', 
            reorder_index[nameTemplate_attn.format(i, 'self_attn', 'out_proj', 'input')]
        )

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def add_act_quant_wrapper_opt(model, device, args, scales=None):
    model.config.use_cache = False
    layers = model.model.decoder.layers

    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], OPTDecoderLayer):
            m = QOPTDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QOPTDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)

        m.self_attn.act_quant = partial(quantize_activation_wrapper, args=args)
        m.self_attn.v_quant = partial(quantize_attn_v_wrapper, args=args)
        m.self_attn.k_quant = partial(quantize_attn_k_wrapper, args=args)

        m.fc_act_quant = partial(quantize_activation_wrapper, args=args)
        m.self_attn_layer_norm.act_quant = partial(quantize_activation_wrapper, args=args)
        m.final_layer_norm.act_quant = partial(quantize_activation_wrapper, args=args)
        
        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model_opt(model, device, args):
    model.config.use_cache = False
    layers = model.model.decoder.layers
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], OPTDecoderLayer):
            m = QOPTDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QOPTDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)
        m.fc1.quant()
        m.fc2.quant()
        m.self_attn.k_proj.quant()
        m.self_attn.v_proj.quant()
        m.self_attn.q_proj.quant()
        m.self_attn.out_proj.quant()

        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model_gptq_opt(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
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
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    quantizers = {}
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], OPTDecoderLayer):
            m = QOPTDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QOPTDecoderLayer):
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
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer, m
        torch.cuda.empty_cache()
        gc.collect()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model