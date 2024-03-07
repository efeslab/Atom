import torch
import torch.nn as nn
from quant import Quantizer
from qLinearLayer import QLinearLayer
from quant import quantize_tensor_channel_group, quantize_tensor, Quantizer
from svd_llama import ASVDLinear

def find_qSVDLinear_layers(module, name=''):
    if type(module) == QSVDLinearLayer:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qSVDLinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class QSVDLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: ASVDLinear,
        args
    ):
        super().__init__()
        self.args = args
        self.register_buffer('Aweight', originalLayer.ALinear.weight)
        self.register_buffer('Bweight', originalLayer.BLinear.weight)
        if originalLayer.ALinear.bias is not None:
            self.register_buffer("Abias", originalLayer.ALinear.bias)
        else:
            self.Abias = None

        #self.act_quant = None
        self.act_quant = Quantizer(args=args)
        
    def to(self, *args, **kwargs):
        super(QSVDLinearLayer, self).to(*args, **kwargs)
        self.Aweight = self.Aweight.to(*args, **kwargs)
        self.Bweight = self.Bweight.to(*args, **kwargs)
        #self.act_quant = self.act_quant.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized
        tmpResult = torch.functional.F.linear(x, self.Bweight)
        tmpResult = self.act_quant(tmpResult)
        y = torch.functional.F.linear(tmpResult, self.Aweight, self.Abias)
        return y
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return
    
        # quantized BWeight
        # only do keeping on BWeight
        if self.args.keeper > 0:
            saved_bw = self.Bweight[:, -self.args.keeper:].clone().contiguous()
            saved_bw2 = self.Bweight[-128:, :-self.args.keeper].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 3:
                saved_bw = quantize_tensor(saved_bw, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)
                saved_bw2 = quantize_tensor(saved_bw2, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)
        
        if self.args.keeper > 0:
            self.Bweight[:, -self.args.keeper:] = 0
            self.Bweight[-128:, :-self.args.keeper] = 0

        
        self.Bweight = quantize_tensor_channel_group(
            self.Bweight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling
        )
        
        if self.args.keeper > 0:
            self.Bweight[:, -self.args.keeper:] = saved_bw
            self.Bweight[-128:, :-self.args.keeper] = saved_bw2
            del saved_bw
            del saved_bw2
        
        if self.args.keeper > 0:
            saved_aw = self.Aweight[:, -128:].clone().contiguous()
            if self.args.keeper_precision > 0:
                saved_aw = quantize_tensor(saved_aw, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)
            self.Aweight[:, -128:] = 0
        
        # quantized AWeight
        self.Aweight = quantize_tensor_channel_group(
            self.Aweight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling
        )
        if self.args.keeper > 0:
            self.Aweight[:, -128:] = saved_aw
            del saved_aw
        
    
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.Bweight.device)
            self.Bweight = torch.index_select(self.Bweight, 1, in_reorder_index)
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.Aweight.device)
                self.Aweight = torch.index_select(self.Aweight, 0, out_reorder_index)
    
    
    def reorder_rank_dim(self, reorder_index):
        if self.args.reorder == True:
            reorder_index = reorder_index.to(self.Bweight.device)
            self.Bweight = torch.index_select(self.Bweight, 0, reorder_index)
            self.Aweight = torch.index_select(self.Aweight, 1, reorder_index)
                
                
                
    def configure(self, func, scales):
        self.act_quant.configure(func, scales)
        # if self.args.static == False:
        #     self.act_quant = func
        #     return
        # else:
        #     assert False, "No implementation error"
        
            