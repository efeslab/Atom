import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
import warnings
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralSparseMoeBlock, MixtralRMSNorm, MixtralAttention, MixtralBlockSparseTop2MLP
from transformers.cache_utils import Cache
from quant import Quantizer
from qLinearLayer import QLinearLayer

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

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


class QMixtralRMSNorm(nn.Module):
    def __init__(
        self,
        originalRMSNorm: MixtralRMSNorm,
        args
    ):
        super().__init__()
        self.originalRMSNorm = originalRMSNorm
        self.register_buffer("reorder_index", None)
        self.args = args

    @torch.no_grad()
    def forward(self, hidden_states):
        result = self.originalRMSNorm(hidden_states)
        if self.reorder_index is not None:
            assert result.shape[result.dim()-1] == self.reorder_index.shape[0]
            result = torch.index_select(result, result.dim()-1, self.reorder_index)

        return result
    
    def to(self, *args, **kwargs):
        super(QMixtralRMSNorm, self).to(*args, **kwargs)
        self.originalRMSNorm = self.originalRMSNorm.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self


class QMixtralAttention(nn.Module):
    def __init__(
            self, 
            originalAttn: MixtralAttention,
            args
        ):
        super().__init__()
        self.config = originalAttn.config
        self.layer_idx = originalAttn.layer_idx

        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = originalAttn.head_dim 
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta
        self.is_causal = True
        self.attention_dropout = originalAttn.attention_dropout
        self.register_buffer("reorder_index", None)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = QLinearLayer(originalAttn.q_proj, args)
        self.k_proj = QLinearLayer(originalAttn.k_proj, args)
        self.v_proj = QLinearLayer(originalAttn.v_proj, args)
        self.o_proj = QLinearLayer(originalAttn.o_proj, args)

        self.rotary_emb = originalAttn.rotary_emb

        self.act_quant = lambda x: x
        self.k_quant = lambda x: x
        self.v_quant = lambda x: x

        self.q_kv_cache = args.kv_cache

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QMixtralAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs)

        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = self.k_quant(key_states)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Fake quantize the value_states
        if self.q_kv_cache:
            value_states = self.v_quant(value_states)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
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
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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

        attn_output = self.act_quant(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class QMixtralBlockSparseTop2MLP(nn.Module):
    def __init__(
        self, 
        originalTop2MLP: MixtralBlockSparseTop2MLP,
        args
    ):
        super().__init__()
        self.ffm_dim = originalTop2MLP.ffn_dim
        self.hidden_dim = originalTop2MLP.hidden_dim

        self.w1 = QLinearLayer(originalTop2MLP.w1, args)
        self.w2 = QLinearLayer(originalTop2MLP.w2, args)
        self.w3 = QLinearLayer(originalTop2MLP.w3, args)

        self.act_fn = originalTop2MLP.act_fn
        self.act_quant = lambda x: x

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.act_quant(current_hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    def quant(self):
        self.w1.quant()
        self.w2.quant()
        self.w3.quant()
        return

    def to(self, *args, **kwargs):
        super(QMixtralBlockSparseTop2MLP, self).to(*args, **kwargs)
        self.w1 = self.w1.to(*args, **kwargs)
        self.w2 = self.w2.to(*args, **kwargs)
        self.w3 = self.w3.to(*args, **kwargs)
        return self


class QMixtralSparseMoeBlock(nn.Module):
    def __init__(
        self,
        originalMoeBlock: MixtralSparseMoeBlock,
        args
    ):
        super().__init__()
        self.hidden_dim = originalMoeBlock.hidden_dim
        self.ffn_dim = originalMoeBlock.ffn_dim
        self.num_experts = originalMoeBlock.num_experts
        self.top_k = originalMoeBlock.top_k
        self.args = args
        self.act_quant = lambda x: x

        # gating
        self.gate = QLinearLayer(originalMoeBlock.gate, args, enable_quant=False)

        self.experts = nn.ModuleList(
            [QMixtralBlockSparseTop2MLP(originalMoeBlock.experts[i], args) for i in range(self.num_experts)]
        )

    def to(self, *args, **kwargs):
        super(QMixtralSparseMoeBlock, self).to(*args, **kwargs)
        self.gate = self.gate.to(*args, **kwargs)
        self.experts = self.experts.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # quantize activations after the MoE gate
        if self.args.abits < 16:
            hidden_states = self.act_quant(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QMixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: MixtralDecoderLayer,
        args
    ):
        super().__init__()
        self.args = args
        self.hidden_size = originalLayer.hidden_size
        self.act_quant = lambda x: x

        self.self_attn = QMixtralAttention(originalLayer.self_attn, args)

        self.block_sparse_moe = QMixtralSparseMoeBlock(originalLayer.block_sparse_moe, args)
        self.input_layernorm = QMixtralRMSNorm(originalLayer.input_layernorm, args)
        self.post_attention_layernorm = QMixtralRMSNorm(originalLayer.post_attention_layernorm, args)

    def to(self, *args, **kwargs):
        super(QMixtralDecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.block_sparse_moe = self.block_sparse_moe.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # quantize activations before feed it to the attention module
        if self.args.abits < 16:
            hidden_states = self.act_quant(hidden_states)

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
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs