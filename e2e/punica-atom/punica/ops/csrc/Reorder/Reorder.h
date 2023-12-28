#include <cuda_fp16.h>

template<int group_size, int hidden_dim>
void run_reorder_fp16_i4(
  half *hidden_states, 
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
);