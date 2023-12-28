#include <cstdint>
#include <cstddef>

template<typename T>
bool DenseLayerGEMM_i4(
  const uint8_t *A,
  const uint8_t *B,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  const uint8_t *A_keeper,
  const uint8_t *B_keeper,
  const uint8_t *A_keeper_scale,
  const uint8_t *B_keeper_scale,
  T *D,
  const size_t M_GLOBAL,
  const size_t N_GLOBAL,
  const size_t K_GLOBAL
);
