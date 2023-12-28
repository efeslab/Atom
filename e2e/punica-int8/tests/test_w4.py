import pytest
import torch

import punica.ops

@torch.inference_mode()
def test_w4_gemv():
    torch.manual_seed(0xABCDABCD987)
    h = 4096
    bs = 7
    device = torch.device("cuda:0")
    group_size = 128

    x = torch.randn((bs, h), device=device, dtype=torch.float16)
    w = torch.randint(16, 64, (h, h // 8), device=device, dtype=torch.int32)

    scales = torch.randn((h, h // group_size), device=device, dtype=torch.float16)
    zeros = torch.zeros((h, h // group_size // 8), device=device, dtype=torch.int32)

    print("test_w4_gemv")
    print(punica.ops.gemv_forward(x, w, scales, zeros))

@torch.inference_mode()
def test_w4_gemm():
    torch.manual_seed(0xABCDABCD987)
    h = 4096
    bs = 177
    device = torch.device("cuda:0")
    group_size = 128

    x = torch.randn((bs, h), device=device, dtype=torch.float16)
    w = torch.randint(16, 64, (h, h // 8), device=device, dtype=torch.int32)

    scales = torch.randn((h, h // group_size), device=device, dtype=torch.float16)
    zeros = torch.zeros((h, h // group_size // 8), device=device, dtype=torch.int32)

    print("test_w4_gemm")
    print(punica.ops.gemm_forward(x, w, scales, zeros))