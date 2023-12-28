#pragma once
#include <cuda_fp16.h>
#include <cstdint>

template<int group_size, int hidden_dim>
void run_activate_fp16_i4(
  nv_half *A,
  nv_half *B,
  int seq_len,
  int8_t *o_outliers,
  int8_t *o_norms,
  nv_half *outlier_scales,
  nv_half *norm_scales
);
