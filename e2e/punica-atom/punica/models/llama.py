# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math
from typing import Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (LlamaConfig, LlamaRMSNorm,
                                                      PreTrainedModel,
                                                      rotate_half)

import punica.ops
from punica.utils import BatchedKvCacheInt4, BatchLenInfo


def rotary_pos_emb(q, k, beg):
  device = q.device
  dtype = q.dtype
  bsz, nhead, seqlen, dim = q.shape
  end = beg + seqlen

  base = 10000
  inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float().to(device) / dim))
  t = torch.arange(beg, end, device=device, dtype=dtype)
  freqs = torch.einsum("i,j->ij", t, inv_freq)
  emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)
  cos = emb.cos()
  sin = emb.sin()
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed.to(q.dtype), k_embed.to(k.dtype)


class LinearInt4(nn.Module):

  def __init__(self, in_features, out_features, out_dtype, bias=False):
    super().__init__()
    assert bias is False
    self.in_features = in_features
    self.out_features = out_features
    self.out_dtype = out_dtype
    group_size = 128
    self.weight_int4 = nn.Parameter(
        torch.empty(
            out_features, (in_features - group_size) // 2, dtype=torch.uint8),
        requires_grad=False)
    self.weight_int8 = nn.Parameter(
        torch.empty(out_features, group_size, dtype=torch.int8),
        requires_grad=False)
    self.scale_int4 = nn.Parameter(
        torch.empty((in_features // group_size - 1,
                     punica.ops.scale_size(out_features)),
                    dtype=torch.float16),
        requires_grad=False)
    self.scale_int8 = nn.Parameter(
        torch.empty(punica.ops.scale_size(out_features), dtype=torch.float16),
        requires_grad=False)
    self.register_parameter("bias", None)

  def forward(self, input):
    outlier, norms, outlier_scales, norm_scales = input
    f = {
        "int4": punica.ops.dense_layer_gemm_i4_o4,
        "fp16": punica.ops.dense_layer_gemm_i4_fp16,
    }[self.out_dtype]
    return f(norms, self.weight_int4, norm_scales, self.scale_int4, outlier,
             self.weight_int8, outlier_scales, self.scale_int8)


class LlamaMLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = LinearInt4(
        self.hidden_size, self.intermediate_size, out_dtype="fp16", bias=False)
    self.up_proj = LinearInt4(
        self.hidden_size, self.intermediate_size, out_dtype="fp16", bias=False)
    self.down_proj = LinearInt4(
        self.intermediate_size, self.hidden_size, out_dtype="fp16", bias=False)

  def forward(self, x):
    return self.down_proj(
        punica.ops.activate_fp16_i4(self.gate_proj(x), self.up_proj(x)))


class LlamaAttention(nn.Module):

  def __init__(self, config: LlamaConfig, layer_idx: int):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self._scale = 1 / math.sqrt(self.head_dim)
    self.layer_idx = layer_idx

    if (self.head_dim * self.num_heads) != self.hidden_size:
      raise ValueError(
          f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
          f" and `num_heads`: {self.num_heads}).")
    self.q_proj = LinearInt4(
        self.hidden_size,
        self.num_heads * self.head_dim,
        out_dtype="fp16",
        bias=False)
    self.k_proj = LinearInt4(
        self.hidden_size,
        self.num_heads * self.head_dim,
        out_dtype="int4",
        bias=False)
    self.v_proj = LinearInt4(
        self.hidden_size,
        self.num_heads * self.head_dim,
        out_dtype="int4",
        bias=False)
    self.o_proj = LinearInt4(
        self.num_heads * self.head_dim,
        self.hidden_size,
        out_dtype="fp16",
        bias=False)
    self.reorder_index = nn.Parameter(
        torch.randperm(self.hidden_size, dtype=torch.int16),
        requires_grad=False)

  def forward(
      self,
      hidden_states,
      blen: BatchLenInfo,
      prefill_kv: BatchedKvCacheInt4 | None,
      decode_kv: BatchedKvCacheInt4 | None,
  ) -> torch.Tensor:
    torch.cuda.nvtx.range_push("qkv_proj")
    q_proj = self.q_proj(hidden_states)
    k_proj = self.k_proj(hidden_states)
    v_proj = self.v_proj(hidden_states)
    torch.cuda.nvtx.range_pop()
    stack_attn_output = []

    if len(blen.prefills) > 0:
      torch.cuda.nvtx.range_push("init_kv")
      assert prefill_kv is not None
      punica.ops.init_kv_i4(
          prefill_kv,
          k_proj[0][:blen.doff].view(-1, self.num_heads, self.head_dim // 2),
          v_proj[0][:blen.doff].view(-1, self.num_heads, self.head_dim // 2),
          k_proj[1][:blen.doff].view(-1, self.num_heads,
                                     self.head_dim // 128 * 2),
          v_proj[1][:blen.doff].view(-1, self.num_heads,
                                     self.head_dim // 128 * 2),
          blen.indptr,
          self.layer_idx,
      )
      torch.cuda.nvtx.range_pop()

      q_projs = q_proj[:blen.doff].split(blen.prefills)
      # k_projs = k_proj[:blen.doff].split(blen.prefills)
      # v_projs = v_proj[:blen.doff].split(blen.prefills)
      for batch_idx, q_len in enumerate(blen.prefills):
        torch.cuda.nvtx.range_push(f"batch_idx={batch_idx}")
        torch.cuda.nvtx.range_push("transpose")
        query_states = q_projs[batch_idx].view(1, q_len, self.num_heads,
                                               self.head_dim).transpose(1, 2)
        # key_states = k_projs[batch_idx].view(1, q_len, self.num_heads,
        #                                      self.head_dim).transpose(1, 2)
        # value_states = v_projs[batch_idx].view(1, q_len, self.num_heads,
        #                                        self.head_dim).transpose(1, 2)
        # HACK:
        query_states = query_states.contiguous()
        key_states = torch.randn_like(query_states)
        value_states = torch.randn_like(query_states)
        # (1, n, s, d)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("pos_emb")
        query_states, key_states = rotary_pos_emb(query_states, key_states, 0)
        torch.cuda.nvtx.range_pop()

        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        # (n, s, d)

        # scaled dot product attention
        torch.cuda.nvtx.range_push("sdpa")
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(q_len, self.hidden_size)
        stack_attn_output.append(attn_output)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    if blen.decode > 0:
      q = q_proj[blen.doff:].view(blen.decode, self.num_heads, self.head_dim)
      k = k_proj[0][blen.doff:].view(blen.decode, self.num_heads,
                                     self.head_dim // 2)
      v = v_proj[0][blen.doff:].view(blen.decode, self.num_heads,
                                     self.head_dim // 2)
      ks = k_proj[1][blen.doff:].view(blen.decode, self.num_heads,
                                      self.head_dim // 128 * 2)
      vs = v_proj[1][blen.doff:].view(blen.decode, self.num_heads,
                                      self.head_dim // 128 * 2)

      torch.cuda.nvtx.range_push("append_kv")
      assert decode_kv is not None
      punica.ops.append_kv_i4(decode_kv, k, v, ks, vs, self.layer_idx)
      torch.cuda.nvtx.range_pop()

      torch.cuda.nvtx.range_push(f"batch_decode")
      attn_outputs = punica.ops.batch_decode_i4(q, decode_kv, self.layer_idx)
      attn_outputs = attn_outputs.view(blen.decode, self.hidden_size)
      stack_attn_output.append(attn_outputs)
      torch.cuda.nvtx.range_pop()

    if len(stack_attn_output) == 1:
      attn_outputs = stack_attn_output[0]
    else:
      attn_outputs = torch.cat(stack_attn_output, dim=0)

    # output projection
    torch.cuda.nvtx.range_push("o_proj")
    reordered = punica.ops.reorder_fp16_i4(attn_outputs, self.reorder_index)
    attn_output = self.o_proj(reordered)
    torch.cuda.nvtx.range_pop()

    return attn_output


class LlamaRMSNormInt4(nn.Module):

  def __init__(self, hidden_size, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.reorder_index = nn.Parameter(
        torch.randperm(hidden_size, dtype=torch.int16), requires_grad=False)

  def forward(self, hidden_states):
    return punica.ops.rmsnorm_fp16_i4(hidden_states, self.weight,
                                      self.reorder_index, self.variance_epsilon)


class LlamaDecoderLayer(nn.Module):

  def __init__(self, config: LlamaConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
    self.mlp = LlamaMLP(config)
    self.input_layernorm = LlamaRMSNormInt4(
        config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = LlamaRMSNormInt4(
        config.hidden_size, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      blen: BatchLenInfo,
      prefill_kv: BatchedKvCacheInt4 | None,
      decode_kv: BatchedKvCacheInt4 | None,
  ) -> torch.Tensor:
    residual = hidden_states

    torch.cuda.nvtx.range_push("input_norm")
    hidden_states = self.input_layernorm(hidden_states)
    torch.cuda.nvtx.range_pop()

    # Self Attention
    torch.cuda.nvtx.range_push("LlamaAttention")
    hidden_states = self.self_attn(hidden_states, blen, prefill_kv, decode_kv)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("r")
    hidden_states = residual + hidden_states
    torch.cuda.nvtx.range_pop()

    # Fully Connected
    residual = hidden_states
    torch.cuda.nvtx.range_push("norm")
    hidden_states = self.post_attention_layernorm(hidden_states)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("mlp")
    hidden_states = self.mlp(hidden_states)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("r")
    hidden_states = residual + hidden_states
    torch.cuda.nvtx.range_pop()

    return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
  config_class = LlamaConfig
  base_model_prefix = "model"
  supports_gradient_checkpointing = False
  _no_split_modules = ["LlamaDecoderLayer"]
  _keys_to_ignore_on_load_unexpected = [
      r"decoder\.version",
      r"self_attn\.rotary_emb\.inv_freq",
  ]


class LlamaModel(LlamaPreTrainedModel):

  def __init__(self, config: LlamaConfig):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                     self.padding_idx)
    self.layers = nn.ModuleList(
        # [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        [LlamaDecoderLayer(config, 0) for _ in range(config.num_hidden_layers)]) # Hack for memory
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_init()

  def forward(
      self,
      input_ids: torch.Tensor,
      blen: BatchLenInfo,
      prefill_kv: BatchedKvCacheInt4 | None,
      decode_kv: BatchedKvCacheInt4 | None,
  ) -> torch.Tensor:
    torch.cuda.nvtx.range_push(f"embed")
    hidden_states = self.embed_tokens(input_ids)
    torch.cuda.nvtx.range_pop()

    for layer_idx, decoder_layer in enumerate(self.layers):
      torch.cuda.nvtx.range_push(f"layer={layer_idx}")
      hidden_states = decoder_layer(hidden_states, blen, prefill_kv, decode_kv)
      torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("lastnorm")
    hidden_states = self.norm(hidden_states)
    torch.cuda.nvtx.range_pop()

    return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    self.model = LlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()

  def forward(
      self,
      input_ids: torch.Tensor,
      blen: BatchLenInfo,
      prefill_kv: BatchedKvCacheInt4 | None,
      decode_kv: BatchedKvCacheInt4 | None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.nvtx.range_push("LlamaForCausalLM")
    hidden_states = self.model(input_ids, blen, prefill_kv, decode_kv)
    torch.cuda.nvtx.range_push("lm_head")
    logits = self.lm_head(hidden_states)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    return logits, hidden_states
