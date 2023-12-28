#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>

bool DenseLayerGEMM_i4_o4(
  const uint8_t *A,
  const uint8_t *B,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  const uint8_t *A_keeper,
  const uint8_t *B_keeper,
  const uint8_t *A_keeper_scale,
  const uint8_t *B_keeper_scale,
  uint8_t *D,
  const size_t M_GLOBAL,
  const size_t N_GLOBAL,
  const size_t K_GLOBAL,
  nv_half2 * output_scale
);
