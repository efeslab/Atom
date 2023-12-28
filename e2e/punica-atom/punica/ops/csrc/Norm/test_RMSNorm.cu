#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

#include "RMSNorm.cu"

/// RUN: nvcc -arch=sm_89 -std=c++17 test_RMSNormQ.cu -o test_RMSNormQ &&
/// ./test_RMSNormQ

#define CUDA_CHECK(status)                                                \
{                                                                         \
    cudaError_t error = status;                                           \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                    << " at line: " << __LINE__ << std::endl;             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

size_t nearest_power2(size_t v) {
  if (v == 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

// void run_rmsnorm_fp16_i4(half *hidden_states, half *weight, float eps,
//                          int group_size, int hidden_dim, int seq_len,
//                          int16_t *reorder_index, int8_t *o_outliers,
//                          int8_t *o_norms, half *outlier_scales,
//                          half *norm_scales) {

//   /// Launch config
//   dim3 grids(seq_len);
//   int elements_per_thread = group_size;
//   assert(hidden_dim % elements_per_thread == 0 &&
//          "Hidden dimension should be multiple of 32.");
//   assert(group_size % elements_per_thread == 0 &&
//          "Group size should be multiple of 32.");
//   int threads = min(cdiv(hidden_dim, elements_per_thread),
//                     1024); // at most use 1024 threads
//   threads = nearest_power2((size_t)threads);
//   if (threads < 32) {
//     threads = 32;
//   }
//   dim3 blocks(threads);
//   int threads_per_group = cdiv(group_size, elements_per_thread);
//   assert(threads_per_group <= 32 &&
//          "Don's support if one warp can't handle one group.");
//   size_t smem_size = hidden_dim * sizeof(half) +
//                      mymax(threads * sizeof(float), hidden_dim * sizeof(half));
//   //   smem_size = mymax(smem_size, 2*1024);
//   std::cout << "Threads: " << threads << "\n";
//   std::cout << "Blocks: " << seq_len << "\n";
//   std::cout << "Smem Size: " << smem_size / 1024.0 << " KB\n";
//   std::cout << "Threads Per Group: " << threads_per_group << "\n";

// #define INVOKE(T, G)                                                           \
//   rmsnorm_fp16_i4_kernel<T, G, 4096><<<grids, blocks, smem_size, nullptr>>>(         \
//       (half *)hidden_states, (half *)weight, eps,           \
//       (int16_t *)reorder_index, (int8_t *)o_outliers,              \
//       (int8_t *)o_norms, (half *)outlier_scales, (half *)norm_scales, SCALE_SIZE_A(seq_len));

// #define SWITCH_T(G)                                                            \
//   switch (threads) {                                                           \
//   case 1024:                                                                   \
//     INVOKE(1024, G)                                                            \
//     break;                                                                     \
//   case 512:                                                                    \
//     INVOKE(512, G)                                                             \
//     break;                                                                     \
//   case 256:                                                                    \
//     INVOKE(256, G)                                                             \
//     break;                                                                     \
//   case 128:                                                                    \
//     INVOKE(128, G)                                                             \
//     break;                                                                     \
//   case 64:                                                                     \
//     INVOKE(64, G)                                                              \
//     break;                                                                     \
//   case 32:                                                                     \
//     INVOKE(32, G)                                                              \
//     break;                                                                     \
//   }

//   switch (group_size) {
//   case 32:
//     SWITCH_T(32);
//     break;
//   case 64:
//     SWITCH_T(64);
//     break;
//   case 128:
//     SWITCH_T(128);
//     break;
//   case 256:
//     SWITCH_T(256);
//     break;
//   }
//   // Check for errors
//   cudaError_t errors = cudaGetLastError();
//   if (errors != cudaSuccess) {
//     printf("rmsnorm_f16_i4_kernel errors: %s\n", cudaGetErrorString(errors));
//     exit(-1);
//   }
// }

void run_cpu_rmsnorm_fp16_i4(half *hidden_states, half *weight, float eps,
                             int group_size, int hidden_dim, int seq_len,
                             int16_t *reorder_index, int8_t *o_outliers,
                             int8_t *o_norms, half *outlier_scales,
                             half *norm_scales) {
  int num_groups = hidden_dim / group_size;
  PackInt4 *int4_o_norms = reinterpret_cast<PackInt4 *>(o_norms);
  float *tmp_states = (float *)malloc(hidden_dim * sizeof(float));
  float *low_bits = (float *)malloc((group_size) * sizeof(float));
  float *high_bits = (float *)malloc(group_size * sizeof(float));
  for (int row = 0; row < seq_len; ++row) {
    float sum_v = 0.0;
    for (int i = 0; i < hidden_dim; ++i) {
      sum_v += (float)hidden_states[row * hidden_dim + i] *
               (float)hidden_states[row * hidden_dim + i];
    }
    sum_v /= hidden_dim;
    sum_v = 1 / std::sqrt(sum_v + eps);
    for (int i = 0; i < hidden_dim; ++i) {
      tmp_states[i] =
          (float)hidden_states[row * hidden_dim + i] * sum_v * (float)weight[i];
    }
    // INT4
    for (int g = 0; g < num_groups - 1; ++g) {
      for (int i = 0; i < group_size; ++i) {
        low_bits[i] = tmp_states[reorder_index[g * group_size + i]];
      }
      float low_bits_max = abs((float)low_bits[0]);
      for (int i = 0; i < group_size; ++i) {
        if (abs((float)low_bits[i]) > low_bits_max) {
          low_bits_max = abs((float)low_bits[i]);
        }
      }
      float low_bit_scale = low_bits_max / 7;
      for (int i = 0; i < group_size; i += 2) {
        int4_o_norms[row * (hidden_dim - group_size) / 2 +
                     (g * group_size + i) / 2]
            .low =
            (int8_t)clamp((int)std::round(low_bits[i] / low_bit_scale), -8, 7);
        int4_o_norms[row * (hidden_dim - group_size) / 2 +
                     (g * group_size + i) / 2]
            .high = (int8_t)clamp(
            (int)std::round(low_bits[i + 1] / low_bit_scale), -8, 7);
      }
      int replicated_row = scale_index(row);
      for(int k = 0; k < 4;++k){
        norm_scales[g * SCALE_SIZE_A(seq_len) + replicated_row + 2 * k] = (half)low_bit_scale;
      }
    }

    // INT8
    for (int i = 0; i < group_size; ++i) {
      high_bits[i] = tmp_states[reorder_index[i + hidden_dim - group_size]];
    }
    float high_bits_max = abs(high_bits[0]);
    for (int i = 0; i < group_size; ++i) {
      if (abs((float)high_bits[i]) > high_bits_max) {
        high_bits_max = abs(high_bits[i]);
      }
    }
    float high_bit_scale = high_bits_max / 127;
    for (int i = 0; i < group_size; i += 1) {
      o_outliers[row * group_size + i] = (int8_t)clamp(
          (int)std::round(high_bits[i] / high_bit_scale), -128, 127);
    }
    int replicated_row = scale_index(row);
    for(int k = 0;k < 4; ++k){
      outlier_scales[replicated_row + 2 * k] = (half)high_bit_scale;
    }
  }
  free(tmp_states);
  free(low_bits);
  free(high_bits);
}

void perf_gpu(half *hidden_states, half *weight, float eps, int group_size,
              int hidden_dim, int seq_len, int16_t *reorder_index,
              int8_t *o_outliers, int8_t *o_norms, half *outlier_scales,
              half *norm_scales, int iters = 200) {
  /// Launch config
  dim3 grids(seq_len);
  int elements_per_thread = group_size;
  assert(hidden_dim % elements_per_thread == 0 &&
         "Hidden dimension should be multiple of 32.");
  assert(group_size % elements_per_thread == 0 &&
         "Group size should be multiple of 32.");
  int threads = min(cdiv(hidden_dim, elements_per_thread),
                    1024); // at most use 1024 threads
  threads = nearest_power2((size_t)threads);
  if (threads < 32) {
    threads = 32;
  }
  dim3 blocks(threads);
  int threads_per_group = cdiv(group_size, elements_per_thread);
  assert(threads_per_group <= 32 &&
         "Don's support if one warp can't handle one group.");
  size_t smem_size = hidden_dim * sizeof(half);
  std::cout << "Threads: " << threads << "\n";
  std::cout << "Blocks: " << seq_len << "\n";
  std::cout << "Smem Size: " << smem_size / 1024.0 << " KB\n";
  std::cout << "Threads Per Group: " << threads_per_group << "\n";

#define INVOKE(T, G)                                                           \
  rmsnorm_fp16_i4_kernel<T, G, 4096><<<grids, blocks, smem_size, nullptr>>>(         \
      (half *)hidden_states, (half *)weight, eps,           \
      (int16_t *)reorder_index, (int8_t *)o_outliers,              \
      (int8_t *)o_norms, (half *)outlier_scales, (half *)norm_scales, SCALE_SIZE_A(seq_len));

#define SWITCH_T(G)                                                            \
  switch (threads) {                                                           \
  case 1024:                                                                   \
    INVOKE(1024, G)                                                            \
    break;                                                                     \
  case 512:                                                                    \
    INVOKE(512, G)                                                             \
    break;                                                                     \
  case 256:                                                                    \
    INVOKE(256, G)                                                             \
    break;                                                                     \
  case 128:                                                                    \
    INVOKE(128, G)                                                             \
    break;                                                                     \
  case 64:                                                                     \
    INVOKE(64, G)                                                              \
    break;                                                                     \
  case 32:                                                                     \
    INVOKE(32, G)                                                              \
    break;                                                                     \
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // warm-up
  for (int i = 0; i < 10; ++i) {
    switch (group_size) {
    case 32:
      SWITCH_T(32);
      break;
    case 64:
      SWITCH_T(64);
      break;
    case 128:
      SWITCH_T(128);
      break;
    case 256:
      SWITCH_T(256);
      break;
    }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    switch (group_size) {
    case 32:
      SWITCH_T(32);
      break;
    case 64:
      SWITCH_T(64);
      break;
    case 128:
      SWITCH_T(128);
      break;
    case 256:
      SWITCH_T(256);
      break;
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "Running cost of CUDA kernel is " << double(ms) / iters
            << "ms\n";
  // Check for errors
  cudaError_t errors = cudaGetLastError();
  if (errors != cudaSuccess) {
    printf("rmsnorm_f16_i4_kernel errors: %s\n", cudaGetErrorString(errors));
    exit(-1);
  }
}

int main() {
  int seq_len = 4096;
  constexpr int hidden_dim = 4096;
  float eps = 1e-5;
  constexpr int group_size = 128;
  int num_groups = hidden_dim / group_size;

  half *hidden_states = (half *)malloc(seq_len * hidden_dim * sizeof(half));
  half *weight = (half *)malloc(hidden_dim * sizeof(half));
  int16_t *reorder_index = (int16_t *)malloc(hidden_dim * sizeof(int16_t));
  int8_t *o_outliers = (int8_t *)malloc(seq_len * group_size * sizeof(int8_t));
  int8_t *o_norms = (int8_t *)malloc(seq_len * (hidden_dim - group_size) / 2 *
                                     sizeof(int8_t));
  half *outlier_scales = (half *)malloc(SCALE_SIZE_A(seq_len) * sizeof(half));
  half *norm_scales = (half *)malloc((num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half));

  int8_t *gold_o_outliers = (int8_t *)malloc(seq_len * group_size * sizeof(int8_t));
  int8_t *gold_o_norms = (int8_t *)malloc(seq_len * (hidden_dim - group_size) / 2 * sizeof(int8_t));

  half *gold_outlier_scales = (half *)malloc(SCALE_SIZE_A(seq_len) * sizeof(half));
  half *gold_norm_scales = (half *)malloc((num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half));

  for (int i = 0; i < seq_len * hidden_dim; ++i) {
    hidden_states[i] = (half)((rand() % 10) * 1.0 / 10);
  }
  for (int i = 0; i < hidden_dim; ++i) {
    weight[i] = (half)((rand() % 10) * 1.0 / 10);
  }
  for (int i = 0; i < hidden_dim; ++i) {
    reorder_index[i] = (int16_t)(i);
  }
  std::random_shuffle(reorder_index, reorder_index + hidden_dim);

  half *dev_hidden_states;
  half *dev_weight;
  int16_t *dev_reorder_index;
  int8_t *dev_o_outliers;
  int8_t *dev_o_norms;
  half *dev_outlier_scales;
  half *dev_norm_scales;

  CUDA_CHECK(cudaMalloc((void **)&dev_hidden_states, seq_len * hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)&dev_weight, hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc((void **)&dev_reorder_index, hidden_dim * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc((void **)&dev_o_outliers,
                        seq_len * group_size * sizeof(int8_t)));
  CUDA_CHECK(
      cudaMalloc((void **)&dev_o_norms,
                 seq_len * (hidden_dim - group_size) / 2 * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc((void **)&dev_outlier_scales, SCALE_SIZE_A(seq_len) * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)&dev_norm_scales, (num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dev_hidden_states, hidden_states,
                        seq_len * hidden_dim * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_weight, weight, hidden_dim * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_reorder_index, reorder_index,
                        hidden_dim * sizeof(int16_t), cudaMemcpyHostToDevice));

  std::cout << "Computing...\n";
  run_rmsnorm_fp16_i4<group_size, hidden_dim>(dev_hidden_states, dev_weight, eps, seq_len, dev_reorder_index, dev_o_outliers,
                      dev_o_norms, dev_outlier_scales, dev_norm_scales);
  std::cout << "Copying results...\n";

  CUDA_CHECK(cudaMemcpy(o_outliers, dev_o_outliers,
                        seq_len * group_size * sizeof(int8_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(o_norms, dev_o_norms,
                 seq_len * (hidden_dim - group_size) / 2 * sizeof(int8_t),
                 cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(outlier_scales, dev_outlier_scales,
                        SCALE_SIZE_A(seq_len) * sizeof(half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(norm_scales, dev_norm_scales,
                        (num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half),
                        cudaMemcpyDeviceToHost));

  std::cout << "Computing golden...\n";
  /// Compute golden
  run_cpu_rmsnorm_fp16_i4(hidden_states, weight, eps, group_size, hidden_dim,
                          seq_len, reorder_index, gold_o_outliers, gold_o_norms,
                          gold_outlier_scales, gold_norm_scales);
  std::cout << "Comparing...\n";

  int errors = 0;
  errors = 0;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < hidden_dim - group_size; j += 2) {
      auto value =
          ((PackInt4 *)o_norms)[i * (hidden_dim - group_size) / 2 + j / 2];
      auto gold =
          ((PackInt4 *)gold_o_norms)[i * (hidden_dim - group_size) / 2 + j / 2];
      if (abs((int)value.low - (int)gold.low) > 1) {
        std::cout << (int)value.low << " vs " << (int)gold.low << "\n";
        errors += 1;
      }
      if (abs((int)value.high - (int)gold.high) > 1) {
        std::cout << (int)value.high << " vs " << (int)gold.high << "\n";
        errors += 1;
      }
    }
  }
  if (errors) {
    std::cout << "Output Norms errors: " << errors << "\n";
  } else {
    std::cout << "Output Norms correct!\n";
  }
  errors = 0;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < group_size; ++j) {
      if (abs(o_outliers[i * group_size + j] -
              gold_o_outliers[i * group_size + j]) > 1) {
        std::cout << (int)o_outliers[i * group_size + j] << " vs "
                  << (int)gold_o_outliers[i * group_size + j] << "\n";
        errors += 1;
      }
    }
  }
  if (errors) {
    std::cout << "Output Outlier errors: " << errors << "\n";
  } else {
    std::cout << "Output Outlier correct!\n";
  }
  errors = 0;
  for (int j = 0; j < num_groups - 1; ++j) {
    for (int i = 0; i < SCALE_SIZE_A(seq_len); ++i) {
      if (abs((float)norm_scales[j * SCALE_SIZE_A(seq_len) + i] -
              (float)gold_norm_scales[j * SCALE_SIZE_A(seq_len) + i]) > 1e-3) {
        std::cout << (float)norm_scales[j * SCALE_SIZE_A(seq_len) + i] << " vs "
                  << (float)gold_norm_scales[j * SCALE_SIZE_A(seq_len) + i] << "\n";
        errors += 1;
      }
    }
  }
  if (errors) {
    std::cout << "Norm Scales errors: " << errors << "\n";
  } else {
    std::cout << "Norm Scales correct!\n";
  }
  errors = 0;
  for (int i = 0; i < SCALE_SIZE_A(seq_len); ++i) {
    if (abs((float)outlier_scales[i] - (float)gold_outlier_scales[i]) > 1e-3) {
      std::cout << (float)outlier_scales[i] << " vs "
                << (float)gold_outlier_scales[i] << "\n";
      errors += 1;
    }
  }
  if (errors) {
    std::cout << "Outlier Scales errors: " << errors << "\n";
  } else {
    std::cout << "Outlier Scales correct!\n";
  }

  // Test performance
  std::cout << "Testing performance...\n";
  perf_gpu(dev_hidden_states, dev_weight, eps, group_size, hidden_dim, seq_len,
           dev_reorder_index, dev_o_outliers, dev_o_norms, dev_outlier_scales,
           dev_norm_scales);

  std::cout << "Done!\n";

  free(hidden_states);
  free(weight);
  free(reorder_index);
  free(o_norms);
  free(o_outliers);
  free(norm_scales);
  free(outlier_scales);
  free(gold_o_norms);
  free(gold_o_outliers);
  free(gold_norm_scales);
  free(gold_outlier_scales);
  CUDA_CHECK(cudaFree(dev_hidden_states));
  CUDA_CHECK(cudaFree(dev_weight));
  CUDA_CHECK(cudaFree(dev_reorder_index));
  CUDA_CHECK(cudaFree(dev_o_norms));
  CUDA_CHECK(cudaFree(dev_o_outliers));
  CUDA_CHECK(cudaFree(dev_norm_scales));
  CUDA_CHECK(cudaFree(dev_outlier_scales));
  return 0;
}