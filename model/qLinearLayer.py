import torch
import torch.nn as nn
from quant import fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args
    ):
        super().__init__()
        self.args = args
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        if self.args.keeper > 0:
            saved_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 1:
                saved_w = fake_quantize_quarter_E5M2(saved_w)
            elif self.args.keeper_precision == 2:
                saved_w = fake_quantize_quarter_E4M3(saved_w)
            elif self.args.keeper_precision == 3:
                saved_w = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = 0

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling
        )

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = saved_w
            del saved_w
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = torch.index_select(self.weight, 1, in_reorder_index)
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = torch.index_select(self.weight, 0, out_reorder_index)