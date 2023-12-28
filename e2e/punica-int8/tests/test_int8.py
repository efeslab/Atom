import pytest
import torch

import torch_int


@torch.inference_mode()
def test_linear_a8_w8_bfp32_ofp32():
    torch.manual_seed(0xABCDABCD987)
    h = 4096
    bs = 7
    device = torch.device("cuda:0")

    x = torch.randint(0, 16, (bs, h), device=device, dtype=torch.int8)
    w = torch.randint(0, 16, (h, h), device=device, dtype=torch.int8)
    bias = torch.randn((h), device=device, dtype=torch.float32)

    print("linear_a8_w8_bfp32_ofp32")
    print(torch_int._CUDA.linear_a8_w8_bfp32_ofp32(x, w, bias, 1.0, 1.0))


@torch.inference_mode()
def test_linear_a8_w8_b8_o8():
    torch.manual_seed(0xABCDABCD987)
    h = 4096
    bs = 7
    device = torch.device("cuda:0")

    x = torch.randint(0, 16, (bs, h), device=device, dtype=torch.int8)
    w = torch.randint(0, 16, (h, h), device=device, dtype=torch.int8)
    bias = torch.randint(0, 16, (h,), device=device, dtype=torch.int8)

    print("linear_a8_w8_b8_o8")
    print(torch_int._CUDA.linear_a8_w8_b8_o8(x, w, bias, 1.0, 1.0))
