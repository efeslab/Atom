"""
This file is a modified version of the original file from the GPTQ repo.
https://github.com/IST-DASLab/gptq
"""
import math
import time
import gc

import torch
import torch.nn as nn
import transformers

from qLlamaLayer import QLinearLayer
from quant import quantize_tensor, fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3
from bitsandbytes.functional import quantize_fp4, dequantize_fp4

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Tool function adopted from GPTQ Codebase
# Modifed for continous channel weight quant and non-uniform quantization
# x: Input Tensor. Most likely x.shape[1] == 1
# scale: Specified scales. Calculated by (2*absmax) / maxq
# zero: Specified zero points. Calculated by -xmin / scale
# maxq: mapped data width
# channel_group: number of channel group quantized together
def quantize_gptq(x, scale, zero, maxq, channel_group, quant_type="int"):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, "only support 2D input when using multilple channel group"
        shape = x.shape
        x = x.reshape((int(x.shape[0]/channel_group), -1))
    # x's layout: [num_groups, group_size]
    if quant_type == "int":
        # Uniform affine mapping
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        q = scale * (q - zero)
    else:
        assert quant_type == "fp", "Currently only support [int, fp]."
        cur_group_size = x.shape[1]
        appended_group_size = 64 - cur_group_size
        assert appended_group_size >= 0, "The least blocksize supported by BNB is 64."
        # first use specified metadata for quantization
        x = torch.clamp(x / scale, -maxq / 2, maxq / 2)
        # Append useless data for ensuring bnb will use scale = 1
        # Basically using bnb quantization kernels as rounding kernel
        x = torch.cat(
            [x,
             torch.ones(x.shape[0], appended_group_size, device=x.device, dtype=x.dtype) * maxq / 2
            ], 
            dim=1
        ).contiguous()
        real_quantize_x, quant_metadata = quantize_fp4(x, blocksize=x.shape[1])
        q = dequantize_fp4(real_quantize_x, quant_metadata)
        # dequantize after rounding
        q = q[:, :cur_group_size].contiguous() * scale
        del real_quantize_x, quant_metadata
    return q.reshape(shape)

class Quantizer_GPTQ(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer_GPTQ, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, channel_group=1, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        clip_ratio=1.0,
        trits=False,
        quant_type="int"
    ):
        if quant_type == "int":
            # Uniform quantization. Width is 2^bits - 1
            self.maxq = torch.tensor(2 ** bits - 1)
        else:
            # Ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
            # [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0]
            assert quant_type == "fp", "Currently only support [int, fp]."
            self.maxq = torch.tensor(2 * 12.0, dtype=torch.float32)

        self.perchannel = perchannel
        self.channel_group = channel_group
        if self.channel_group > 1:
            assert self.perchannel is True, "set perchannel to True when using multilple channel group"
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.clip_ratio = clip_ratio
        self.quant_type = quant_type
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.channel_group > 1:
                    x = x.reshape(int(shape[0]/self.channel_group), -1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
            # shrink the range based on clip ratio
            self.scale = (xmax - xmin) * self.clip_ratio / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize_gptq(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize_gptq(x, self.scale, self.zero, self.maxq, self.channel_group, self.quant_type)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class GPTQ:
    def __init__(self, layer, n_out, keeper_precision=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d): 
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1] 
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) 
        self.nsamples = 0 
        self.keeper_precision = keeper_precision

        self.n_out = n_out
        self.n_nonout = W.shape[1] - n_out
        del W

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] 
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D) or isinstance(self.layer, QLinearLayer):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        assert actorder==False, "we don't deal with actorder inside GPTQ for our implementation."
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
                
        if not self.quantizer.ready():
            self.quantizer.find_params(W[:,:self.n_nonout], weight=True)

        H = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp 
        
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)], weight=True)
                q = quantize_gptq(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero,
                    self.quantizer.maxq, self.quantizer.channel_group, self.quantizer.quant_type
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())
        
        if self.n_out > 0:
            keep_w = W[:,self.n_nonout:].contiguous()

            if self.keeper_precision > 0:
                if self.keeper_precision == 1:
                    keep_w = fake_quantize_quarter_E5M2(keep_w)
                elif self.keeper_precision == 2:
                    keep_w = fake_quantize_quarter_E4M3(keep_w)
                elif self.keeper_precision == 3:
                    keep_w = quantize_tensor(keep_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

            Q[:,self.n_nonout:] = keep_w

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        del H
        del Losses
        del W

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        gc.collect()
