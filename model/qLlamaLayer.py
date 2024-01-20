import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from quant import Quantizer, fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group
from qLinearLayer import QLinearLayer

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Copy from transformers.models.llama.modeling_llama
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class QLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: LlamaDecoderLayer,
        args
    ):
        super().__init__()
        self.args = args
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QLlamaAttention(
            originalLayer.self_attn,
            args
        )
        self.mlp = QLlamaMLP(
            originalLayer.mlp,
            args
        )
        self.input_layernorm = QLlamaRMSNorm(
            originalLayer.input_layernorm, 
            args
        )
        self.post_attention_layernorm = QLlamaRMSNorm(
            originalLayer.post_attention_layernorm, 
            args
        )

    def to(self, *args, **kwargs):
        super(QLlamaDecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class QLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: LlamaRMSNorm,
        args
    ):
        super().__init__()
        self.originalNorm = originalNorm
        self.act_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)
        self.args = args

    @torch.no_grad()
    def forward(self, hidden_states):
        result = self.originalNorm(hidden_states)
        if self.reorder_index is not None:
            assert result.shape[result.dim()-1] == self.reorder_index.shape[0]
            result = torch.index_select(result, result.dim()-1, self.reorder_index)

        if self.args.abits < 16:
            result = self.act_quant(result)

        return result
    
    def to(self, *args, **kwargs):
        super(QLlamaRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

class QLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        originalAttn: LlamaAttention,
        args
    ):
        super().__init__()
        self.abits = args.abits
        self.q_kv_cache = args.kv_cache
        self.config = originalAttn.config
        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = QLinearLayer(
            originalAttn.q_proj,
            args
        )
        self.k_proj = QLinearLayer(
            originalAttn.k_proj,
            args
        )
        self.v_proj = QLinearLayer(
            originalAttn.v_proj,
            args
        )
        self.o_proj = QLinearLayer(
            originalAttn.o_proj,
            args
        )
        self.rotary_emb = originalAttn.rotary_emb
        self.act_quant = Quantizer(args=args)
        self.v_quant = Quantizer(args=args)
        self.k_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QLlamaAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        self.v_quant = self.v_quant.to(*args, **kwargs)
        self.k_quant = self.k_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = self.k_quant(key_states)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Fake quantize the value_states
        if self.q_kv_cache:
            value_states = self.v_quant(value_states)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Reorder the BMM output to feed into o.proj
        if self.reorder_index is not None:
            attn_output = torch.index_select(attn_output, 2, self.reorder_index)

        # Quantize the attention output
        attn_output = self.act_quant(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class QLlamaMLP(nn.Module):
    def __init__(
        self,
        originalMLP: LlamaMLP,
        args
    ):
        super().__init__()
        self.gate_proj = QLinearLayer(
            originalMLP.gate_proj,
            args
        )
        self.down_proj = QLinearLayer(
            originalMLP.down_proj,
            args
        )
        self.up_proj = QLinearLayer(
            originalMLP.up_proj,
            args
        )
        self.act_fn = originalMLP.act_fn
        self.act_quant = Quantizer(args=args)
        # self.register_buffer("act_shifts", None)

    def to(self, *args, **kwargs):
        super(QLlamaMLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Quantize the activations and feed into down_proj
        tmpResult = self.act_quant(tmpResult)
        return self.down_proj(tmpResult)