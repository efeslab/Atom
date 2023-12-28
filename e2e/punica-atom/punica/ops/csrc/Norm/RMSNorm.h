#include <cuda_fp16.h>

template<int group_size, int hidden_dim>
void run_rmsnorm_fp16_i4(
  nv_half *hidden_states,
  nv_half *weight,
  float eps,
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  nv_half *outlier_scales,
  nv_half *norm_scales
);
