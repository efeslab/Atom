import torch
from torch import nn
from functools import partial

# Prototype for tile-wise quantization operator
# from myCuda_functions import tensor_to_blocks, blocks_to_tensor

# Fake round from S1E5M10 to S1E5M2
@torch.no_grad()
def fake_quantize_quarter_E5M2(w: torch.tensor) -> torch.tensor:
    # Sign:     1000 0000 0000 0000 HEX: 0x8000
    # Exponent: 0111 1100 0000 0000 HEX: 0x7C00
    # Mantissa: 0000 0011 1111 1111 HEX: 0x03FF
    assert w.dtype == torch.float16
    w = w.view(torch.int16).cuda()
    # FP16: S1E5M10
    # FP8 : S1E5M2
    mantissa = w & 0x03FF
    roundFloat = (((mantissa << 2) & 0x03FF) + 0x3C00).clone().view(torch.float16).cuda()
    roundingBits = (torch.round(roundFloat) - 1).to(dtype=torch.int16)
    mantissa = ((mantissa >> 8) + roundingBits) << 8

    w = (w & 0xFC00) + mantissa
    return w.view(torch.float16)

# Fake round from S1E5M10 to S1E4M3
def fake_quantize_quarter_E4M3(w: torch.tensor) -> torch.tensor:
    # Ref: https://www.h-schmidt.net/FloatConverter/IEEE754.html
    # Sign:     1000 0000 0000 0000 HEX: 0x8000
    # Exponent: 0111 1100 0000 0000 HEX: 0x7C00
    # Mantissa: 0000 0011 1111 1111 HEX: 0x03FF
    assert w.dtype == torch.float16
    # Maximum number of FP8 E4M3 should be (0 1111 111) = 480
    w = w.cuda()
    w = torch.clamp(w, -480, 480)

    # Manipulate bits
    w = w.view(torch.int16)
    # print in hex
    # First round mantissa
    # Need consider rounding case
    # Just construct a float16 to see whether to round 1 bits
    mantissa = w & 0x03FF
    roundFloat = (((mantissa << 3) & 0x03FF) + 0x3C00).clone().view(torch.float16).cuda()
    roundingBits = (torch.round(roundFloat) - 1).to(dtype=torch.int16)
    mantissa = ((mantissa >> 7) + roundingBits) << 7

    # Deal with subnormal value
    # Round exponent in [-6, 8] + 15 = [9, 23]
    # Ref: https://arxiv.org/pdf/2209.05433.pdf
    # Min normal value:     0 0001 000 = 2^-6
    # Min Submormal Value:  0 0000 001 = 2^-9
    exponent = (w & 0x7C00) >> 10
    subNormalMask = (exponent - 15) < -6
    subNormal_min = torch.tensor(2**(-9), dtype=torch.float16, device='cuda')
    w = w.view(torch.float16)
    w[subNormalMask] = torch.round(w[subNormalMask] / subNormal_min).to(dtype=torch.int16) * subNormal_min
    
    # Deal with normal value
    w = w.view(torch.int16)
    exponent = torch.clamp(exponent, 9, 23)
    exponent = exponent << 10
    w[~subNormalMask] = (w[~subNormalMask] & 0x8000) + mantissa[~subNormalMask] + exponent[~subNormalMask]
    return w.view(torch.float16)


@torch.no_grad()
def quantize_tensor_channel_group(W: torch.tensor, n_bits, group_size, tiling, sym, channel_group=1, clip_ratio=1.0, exponential=False) -> torch.tensor:
    # savedShape = W.shape
    assert W.dim() == 2 or W.shape[0] == 1, "Current only support bsz=1"
    assert group_size == 0 or tiling == 0
    assert W.is_contiguous(), "Input tensor is not contiguous"
    assert n_bits < 16

    # group_size = 0 is per-channel quantization
    if group_size > 0:
        assert W.shape[-1] % group_size == 0

    if group_size == 0:
        W = quantize_tensor(W, n_bits=n_bits, group_size=0, tiling=tiling, sym=sym, exponential=exponential)
    else:
        for i1 in range(0, W.shape[1], group_size):
            i2 = min(i1 + group_size, W.shape[1])
            w = W[:,i1:i2]
            if channel_group > 1:
                w = w.reshape(int(W.shape[0]/channel_group), -1)

            w = quantize_tensor(
                w, n_bits=n_bits, group_size=0, tiling=tiling, sym=sym, clip_ratio=clip_ratio, exponential=exponential
            )

            if channel_group > 1:
                w = w.reshape(-1, group_size)
            W[:,i1:i2] = w

    return W

@torch.no_grad()
def quantize_tensor(w: torch.tensor, n_bits, group_size, tiling, sym, clip_ratio=1.0, exponential=False) -> torch.tensor:
    savedShape = w.shape
    assert w.dim() == 2 or savedShape[0] == 1, "Current only support bsz=1"
    assert group_size == 0 or tiling == 0
    # assert w.is_contiguous(), "Input tensor is not contiguous"

    w = w.squeeze()
    newSavedShape = w.shape # Saved [row, columns]

    if tiling > 0:
        assert tiling == 16, "Currently only support tiling=16"
        assert w.numel() % (tiling**2) == 0, f"shape {w.shape}, {w.numel()} is not divisible by {tiling**2}"
        assert newSavedShape[0] % tiling == 0 and newSavedShape[1] % tiling == 0

        swapTensor = torch.rand(w.numel() // (tiling**2), (tiling**2), device=w.device, dtype=w.dtype)
        tensor_to_blocks(w, swapTensor, tiling)
        w, swapTensor = swapTensor, w

    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format: [-1, group]"
    assert n_bits < 16

    if sym:
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    else:
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)

    if exponential:
        q_max = (2**(2**(n_bits-1)-1))
        q_min = (1)
        if sym:
            scales = w_max
            base = torch.zeros_like(scales)
        else:
            scales = (w_max-w_min) * torch.tensor(0.5)
            base = (w_max+w_min) * torch.tensor(0.5)
        scales.div_(q_max)
        w = w - base
        w_sign = torch.sign(w)
        log_scaled_w = torch.log2((torch.abs(w) / scales).clamp(min=q_min, max=q_max))
        int_scaled_w = torch.floor(log_scaled_w)
        complement = (log_scaled_w - int_scaled_w > torch.log2(torch.tensor(1.5))).int()
        int_scaled_w = int_scaled_w + complement
        w = (2 ** int_scaled_w) * w_sign * scales
        w = w + base
    else: # uniform affine mapping
        if sym:
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            if clip_ratio < 1.0:
                w_max = w_max * clip_ratio
            scales = w_max / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            if clip_ratio < 1.0:
                w_max *= clip_ratio
                w_min *= clip_ratio
            scales = (w_max-w_min).clamp(min=1e-5) / q_max
            base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    if tiling > 0:
        blocks_to_tensor(w, swapTensor, tiling)
        w, swapTensor = swapTensor, w
        del swapTensor
    
    return w.reshape(savedShape)

@torch.no_grad()
def quantize_activation_wrapper(x: torch.tensor, args) -> torch.tensor:
    if args.abits >= 16:
        return x 
    
    qFunction = partial(
        quantize_tensor, 
        n_bits=args.abits, 
        group_size=args.act_group_size, 
        tiling=args.tiling, 
        sym=args.a_sym,
        clip_ratio=args.a_clip_ratio
    )

    savedShape = x.shape
    x = x.view(-1, savedShape[-1])

    assert args.act_group_size == 0 or (savedShape[-1]) % args.act_group_size == 0, "Group size should be divisible by (dim - keeper)."

    if args.keeper > 0:
        saved_x = x[:, -args.keeper:].clone().contiguous()
    
    # Whether to keep outliers in FP8
    if args.keeper and args.keeper_precision > 0:
        assert args.keeper > 0, "Keeper must be greater than 0"
        if args.keeper_precision == 1:
            saved_x = fake_quantize_quarter_E5M2(saved_x)
        elif args.keeper_precision == 2:
            saved_x = fake_quantize_quarter_E4M3(saved_x)
        elif args.keeper_precision == 3:
            saved_x = quantize_tensor(saved_x, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

    if args.keeper > 0:
        x[:, -args.keeper:] = 0
    x = qFunction(x)
    if args.keeper > 0:
        x[:, -args.keeper:] = saved_x
        del saved_x

    return x.view(savedShape)

@torch.no_grad()
def quantize_attn_v_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Input shape: [bsz, self.num_heads, seq_len, self.head_dim]
    # Quantize on head_dim
    assert w.dim() == 4 or  w.dim() == 3
    assert w.shape[-1] == 128
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    # (g128, asym) == (g64, sym)
    # Expressiness is almost equal to bits
    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

@torch.no_grad()
def quantize_attn_k_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Input shape: [bsz, self.num_heads, seq_len, self.head_dim]
    # Quantize on head_dim
    assert w.dim() == 4 or  w.dim() == 3
    assert w.shape[-1] == 128
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

class Quantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        self.act_quant = lambda x: x

    @torch.no_grad()
    def forward(self, hidden_states):
        if self.args.static == False or self.scales is None:
            return self.act_quant(hidden_states)
        
        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == True, "Only support statically symmetric quantization"
        assert self.args.act_group_size == 0 or (savedShape[-1] - self.args.keeper) % self.args.act_group_size == 0, "Group size should be divisible by (dim - keeper)."

        hidden_states = hidden_states.view(-1, savedShape[-1])
        selected_states = hidden_states[:, self.args.keeper:].clone()

        if self.args.act_group_size > 0:
            selected_states = selected_states.reshape(-1, self.args.act_group_size)

        assert self.scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        selected_states = (torch.clamp(torch.round(selected_states / self.scales), self.q_min, self.q_max)) * self.scales
        selected_states = selected_states.reshape(-1, savedShape[-1] - self.args.keeper)
        hidden_states[:, self.args.keeper:] = selected_states

        return hidden_states.view(savedShape)
    
    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        self.q_min = (-2**(self.args.abits-1))
        self.q_max = (2**(self.args.abits-1)-1)