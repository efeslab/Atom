import torch

import punica.ops._kernels
from punica.utils.kvcache import BatchedKvCacheInt4

__all__ = [
    "batch_decode_i4",
    "append_kv_i4",
    "init_kv_i4",
    "bgmv",
    "add_lora",
    "rms_norm",
    "activate_fp16_i4",
    "dense_layer_gemm_i4_fp16",
    "dense_layer_gemm_i4_o4",
    "rmsnorm_fp16_i4",
    "reorder_fp16_i4",
]


def batch_decode_i4(
    q: torch.Tensor,
    kv: BatchedKvCacheInt4,
    layer_idx: int,
):
  device = q.device
  dtype = q.dtype
  o = torch.empty(q.shape, dtype=dtype, device=device)
  punica.ops._kernels.batch_decode_i4(o, q, kv.data, kv.param, kv.indptr,
                                      kv.indicies, kv.last_page_offset,
                                      layer_idx)
  return o


def init_kv_i4(
    kv: BatchedKvCacheInt4,
    k: torch.Tensor,
    v: torch.Tensor,
    k_param: torch.Tensor,
    v_param: torch.Tensor,
    seqlen_indptr: torch.Tensor,
    layer_idx: int,
):
  punica.ops._kernels.init_kv_i4(kv.data, kv.param, kv.indptr, kv.indicies,
                                 kv.last_page_offset, k, v, k_param, v_param,
                                 seqlen_indptr, layer_idx)


def append_kv_i4(
    kv: BatchedKvCacheInt4,
    k: torch.Tensor,
    v: torch.Tensor,
    k_param: torch.Tensor,
    v_param: torch.Tensor,
    layer_idx: int,
):
  punica.ops._kernels.append_kv_i4(kv.data, kv.param, kv.indptr, kv.indicies,
                                   kv.last_page_offset, k, v, k_param, v_param,
                                   layer_idx)


def bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
  """
  Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ w_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
      ).squeeze(0)

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    w_T_all: Shape: `[None, L, H2, H1]`. All of the transposed weight matrices.
    indicies: Shape: `[B]`. Indices of the weight matrices.
    layer_idx: Layer index of the weight matrices.
    scale: Scaling factor.
  """
  f = punica.ops._kernels.dispatch_bgmv
  f(y, x, w_T_all, indicies, layer_idx, scale)


def add_lora(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
  """
  Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
      ).squeeze(0)

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
    wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
    indicies: Shape: `[B]`. Indices of the LoRA weights.
    layer_idx: Layer index of LoRA weights.
    scale: Scaling factor.
  """
  f = punica.ops._kernels.dispatch_bgmv
  device = x.device
  dtype = x.dtype

  r = wb_T_all.size(-1)
  tmp = torch.zeros((x.size(0), r), dtype=dtype, device=device)
  f(tmp, x, wa_T_all, indicies, layer_idx, 1.0)
  f(y, tmp, wb_T_all, indicies, layer_idx, scale)


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
):
  o = torch.empty_like(x)
  punica.ops._kernels.rms_norm(o, x, w, eps)
  return o


def scale_size(x):
  return ((x) // 16 * 64 + 64 - (1 - (x % 16) // 8) * (8 - (x % 8)) * 8)


def activate_fp16_i4(a: torch.Tensor, b: torch.Tensor):
  group_size = 128
  bs, hidden_dim = a.shape
  o_outlier = torch.empty((bs, group_size), dtype=torch.int8, device=a.device)
  o_norms = torch.empty((bs, (hidden_dim - group_size) // 2),
                        dtype=torch.int8,
                        device=a.device)
  outlier_scales = torch.empty((scale_size(bs),),
                               dtype=torch.float16,
                               device=a.device)
  norm_scales = torch.empty((hidden_dim // group_size - 1, scale_size(bs)),
                            dtype=torch.float16,
                            device=a.device)
  punica.ops._kernels.activate_fp16_i4(a, b, bs, o_outlier, o_norms,
                                       outlier_scales, norm_scales)
  return o_outlier, o_norms, outlier_scales, norm_scales


def dense_layer_gemm_i4_fp16(a, b, a_scale, b_scale, a_keeper, b_keeper,
                             a_keeper_scale, b_keeper_scale):
  m = a.size(0)
  n = b.size(0)
  d = torch.empty((m, n), dtype=torch.float16, device=a.device)
  punica.ops._kernels.dense_layer_gemm_i4_fp16(a, b, a_scale, b_scale, a_keeper,
                                               b_keeper, a_keeper_scale,
                                               b_keeper_scale, d)
  return d


def dense_layer_gemm_i4_o4(a, b, a_scale, b_scale, a_keeper, b_keeper,
                           a_keeper_scale, b_keeper_scale):
  m = a.size(0)
  n = b.size(0)
  d = torch.empty((m, n // 2), dtype=torch.uint8, device=a.device)
  assert n % 128 == 0
  d_scale = torch.empty((m, n // 128 * 2), dtype=torch.float16, device=a.device)
  punica.ops._kernels.dense_layer_gemm_i4_o4(a, b, a_scale, b_scale, a_keeper,
                                             b_keeper, a_keeper_scale,
                                             b_keeper_scale, d, d_scale)
  return d, d_scale


def rmsnorm_fp16_i4(hidden_states, weight, reorder_index, eps):
  group_size = 128
  device = hidden_states.device
  bs, hidden_dim = hidden_states.shape
  o_outlier = torch.empty((bs, group_size), dtype=torch.int8, device=device)
  o_norms = torch.empty((bs, (hidden_dim - group_size) // 2),
                        dtype=torch.int8,
                        device=device)
  outlier_scales = torch.empty((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  norm_scales = torch.empty((hidden_dim // group_size - 1, scale_size(bs)),
                            dtype=torch.float16,
                            device=device)
  punica.ops._kernels.rmsnorm_fp16_i4(hidden_states, weight, eps, reorder_index,
                                      o_outlier, o_norms, outlier_scales,
                                      norm_scales)
  return o_outlier, o_norms, outlier_scales, norm_scales


def reorder_fp16_i4(hidden_states, reorder_index):
  group_size = 128
  device = hidden_states.device
  bs, hidden_dim = hidden_states.shape
  o_outlier = torch.empty((bs, group_size), dtype=torch.int8, device=device)
  o_norms = torch.empty((bs, (hidden_dim - group_size) // 2),
                        dtype=torch.int8,
                        device=device)
  outlier_scales = torch.empty((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  norm_scales = torch.empty((hidden_dim // group_size - 1, scale_size(bs)),
                            dtype=torch.float16,
                            device=device)
  punica.ops._kernels.reorder_fp16_i4(hidden_states, reorder_index, o_outlier,
                                      o_norms, outlier_scales, norm_scales)
  return o_outlier, o_norms, outlier_scales, norm_scales
