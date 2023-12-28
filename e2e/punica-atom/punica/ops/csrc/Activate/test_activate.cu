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

#include "Activate.cu"

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

void run_cpu_activate_fp16_i4(
  half *A,
  half *B,
  int group_size,
  int hidden_dim,
  int seq_len,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
) {
  int num_groups = hidden_dim / group_size;
  PackInt4 *int4_o_norms = reinterpret_cast<PackInt4 *>(o_norms);
  float *tmp_states = (float *)malloc(hidden_dim * sizeof(float));
  float *low_bits = (float *)malloc((group_size) * sizeof(float));
  float *high_bits = (float *)malloc(group_size * sizeof(float));
  for (int row = 0; row < seq_len; ++row) {
    for(int i = 0; i < hidden_dim; ++i){
      tmp_states[i] = silu((float)(A[row * hidden_dim + i])) * (float)B[row * hidden_dim + i];
    }
    // INT4
    for (int g = 0; g < num_groups - 1; ++g) {
      for (int i = 0; i < group_size; ++i) {
        low_bits[i] = tmp_states[g * group_size + i];
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
      high_bits[i] = tmp_states[i + hidden_dim - group_size];
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

void perf_gpu(
  half *A,
  half *B,
  int group_size,
  int hidden_dim,
  int seq_len,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales,
  int iters = 200
) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // warm-up
  for (int i = 0; i < 10; ++i) { 
    run_activate_fp16_i4<128, 4096>(
      A,
      B,
      seq_len,
      o_outliers,
      o_norms,
      outlier_scales,
      norm_scales
    );
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    run_activate_fp16_i4<128, 4096>(
      A,
      B,
      seq_len,
      o_outliers,
      o_norms,
      outlier_scales,
      norm_scales
    );
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
  constexpr int group_size = 128;
  int num_groups = hidden_dim / group_size;

  half *A = (half *)malloc(seq_len * hidden_dim * sizeof(half));
  half *B = (half *)malloc(seq_len * hidden_dim * sizeof(half));
  int8_t *o_outliers = (int8_t *)malloc(seq_len * group_size * sizeof(int8_t));
  int8_t *o_norms = (int8_t *)malloc(seq_len * (hidden_dim - group_size) / 2 *
                                     sizeof(int8_t));
  half *outlier_scales = (half *)malloc(SCALE_SIZE_A(seq_len) * sizeof(half));
  half *norm_scales = (half *)malloc((num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half));

  int8_t *gold_o_outliers = (int8_t *)malloc(seq_len * group_size * sizeof(int8_t));
  int8_t *gold_o_norms = (int8_t *)malloc(seq_len * (hidden_dim - group_size) / 2 * sizeof(int8_t));

  half *gold_outlier_scales = (half *)malloc(SCALE_SIZE_A(seq_len) * sizeof(half));
  half *gold_norm_scales = (half *)malloc((num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half));

  // set random seed
  srand(1110);
  for (int i = 0; i < seq_len * hidden_dim; ++i) {
    A[i] = (half)((rand() % 10) * 1.0 / 10);
    B[i] = (half)((rand() % 10) * 1.0 / 10);
  }

  half *dev_A;
  half *dev_B;
  int8_t *dev_o_outliers;
  int8_t *dev_o_norms;
  half *dev_outlier_scales;
  half *dev_norm_scales;

  CUDA_CHECK(cudaMalloc((void **)&dev_A, seq_len * hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)&dev_B, seq_len * hidden_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc((void **)&dev_o_outliers,
                        seq_len * group_size * sizeof(int8_t)));
  CUDA_CHECK(
      cudaMalloc((void **)&dev_o_norms,
                 seq_len * (hidden_dim - group_size) / 2 * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc((void **)&dev_outlier_scales, SCALE_SIZE_A(seq_len) * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)&dev_norm_scales, (num_groups - 1) * SCALE_SIZE_A(seq_len) * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dev_A, A,
                        seq_len * hidden_dim * sizeof(half),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(dev_B, B,
                        seq_len * hidden_dim * sizeof(half),
                        cudaMemcpyHostToDevice));

  std::cout << "Computing...\n";
  run_activate_fp16_i4<group_size, hidden_dim>(
    dev_A,
    dev_B,
    seq_len,
    dev_o_outliers,
    dev_o_norms,
    dev_outlier_scales,
    dev_norm_scales
  );
  cudaDeviceSynchronize();
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
  run_cpu_activate_fp16_i4(
    A,
    B,
    group_size,
    hidden_dim,
    seq_len,
    gold_o_outliers,
    gold_o_norms,
    gold_outlier_scales,
    gold_norm_scales
  );
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
  perf_gpu(
    dev_A,
    dev_B,
    group_size,
    hidden_dim,
    seq_len,
    dev_o_outliers, 
    dev_o_norms,
    dev_outlier_scales,
    dev_norm_scales
  );

  std::cout << "Done!\n";

  free(A);
  free(B);
  free(o_norms);
  free(o_outliers);
  free(norm_scales);
  free(outlier_scales);
  free(gold_o_norms);
  free(gold_o_outliers);
  free(gold_norm_scales);
  free(gold_outlier_scales);
  
  CUDA_CHECK(cudaFree(dev_A));
  CUDA_CHECK(cudaFree(dev_B));
  CUDA_CHECK(cudaFree(dev_o_norms));
  CUDA_CHECK(cudaFree(dev_o_outliers));
  CUDA_CHECK(cudaFree(dev_norm_scales));
  CUDA_CHECK(cudaFree(dev_outlier_scales));
  return 0;
}