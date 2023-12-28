import torch
import punica.ops
from punica.ops import scale_size


@torch.inference_mode()
def test_activate_fp16_i4():
  bs = 7
  hidden_dim = 4096
  dtype = torch.float16
  device = torch.device("cuda:0")
  a = torch.randn((bs, hidden_dim), dtype=dtype, device=device)
  b = torch.randn((bs, hidden_dim), dtype=dtype, device=device)
  print("test_activate_fp16_i4")
  print(punica.ops.activate_fp16_i4(a, b))
  torch.cuda.synchronize()


@torch.inference_mode()
def test_dense_layer_gemm_i4_o4():
  device = torch.device("cuda:0")
  bs = 7
  hidden_dim = 4096
  group_size = 128

  a = torch.randint(
      16,
      128, (bs, (hidden_dim - group_size) // 2),
      dtype=torch.uint8,
      device=device)
  b = torch.randint(
      16,
      128, (hidden_dim, (hidden_dim - group_size) // 2),
      dtype=torch.uint8,
      device=device)
  a_scale = torch.randn((hidden_dim // group_size - 1, scale_size(bs)),
                        dtype=torch.float16,
                        device=device)
  b_scale = torch.randn((hidden_dim // group_size - 1, scale_size(bs)),
                        dtype=torch.float16,
                        device=device)
  a_keeper = torch.randint(
      0, 255, (bs, group_size), dtype=torch.uint8, device=device)
  b_keeper = torch.randint(
      0, 255, (bs, group_size), dtype=torch.uint8, device=device)
  a_keeper_scale = torch.randn((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  b_keeper_scale = torch.randn((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  print("test_dense_layer_gemm_i4_o4")
  print(
      punica.ops.dense_layer_gemm_i4_o4(a, b, a_scale, b_scale, a_keeper,
                                        b_keeper, a_keeper_scale,
                                        b_keeper_scale))
  torch.cuda.synchronize()


@torch.inference_mode()
def test_dense_layer_gemm_i4_fp16():
  device = torch.device("cuda:0")
  bs = 7
  hidden_dim = 4096
  group_size = 128

  a = torch.randint(
      16,
      128, (bs, (hidden_dim - group_size) // 2),
      dtype=torch.uint8,
      device=device)
  b = torch.randint(
      16,
      128, (hidden_dim, (hidden_dim - group_size) // 2),
      dtype=torch.uint8,
      device=device)
  a_scale = torch.randn((hidden_dim // group_size - 1, scale_size(bs)),
                        dtype=torch.float16,
                        device=device)
  b_scale = torch.randn((hidden_dim // group_size - 1, scale_size(bs)),
                        dtype=torch.float16,
                        device=device)
  a_keeper = torch.randint(
      0, 255, (bs, group_size), dtype=torch.uint8, device=device)
  b_keeper = torch.randint(
      0, 255, (bs, group_size), dtype=torch.uint8, device=device)
  a_keeper_scale = torch.randn((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  b_keeper_scale = torch.randn((scale_size(bs),),
                               dtype=torch.float16,
                               device=device)
  print("test_dense_layer_gemm_i4_fp16")
  print(
      punica.ops.dense_layer_gemm_i4_fp16(a, b, a_scale, b_scale, a_keeper,
                                          b_keeper, a_keeper_scale,
                                          b_keeper_scale))
  torch.cuda.synchronize()


@torch.inference_mode()
def test_rmsnorm_fp16_i4():
  device = torch.device("cuda:0")
  bs = 7
  hidden_dim = 4096
  eps = 1e-5
  hidden_states = torch.randn((bs, hidden_dim),
                              dtype=torch.float16,
                              device=device)
  weight = torch.randn((hidden_dim,), dtype=torch.float16, device=device)
  reorder_index = torch.randperm(hidden_dim, dtype=torch.int16, device=device)
  print("test_rmsnorm_fp16_i4")
  print(punica.ops.rmsnorm_fp16_i4(hidden_states, weight, reorder_index, eps))
  torch.cuda.synchronize()


@torch.inference_mode()
def test_reorder_fp16_i4():
  device = torch.device("cuda:0")
  bs = 7
  hidden_dim = 4096
  hidden_states = torch.randn((bs, hidden_dim),
                              dtype=torch.float16,
                              device=device)
  reorder_index = torch.randperm(hidden_dim, dtype=torch.int16, device=device)
  print("test_reorder_fp16_i4")
  print(punica.ops.reorder_fp16_i4(hidden_states, reorder_index))
  torch.cuda.synchronize()
