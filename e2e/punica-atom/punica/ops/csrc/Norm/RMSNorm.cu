#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

namespace cg = cooperative_groups;

struct PackInt4 {
  int8_t low : 4;
  int8_t high : 4;
};

HOST_DEVICE int cdiv(int a, int b) { return (a + b - 1) / b; }

HOST_DEVICE int clamp(int x, int a, int b) { return max(a, min(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
HOST_DEVICE Accum local_sum_p2(U *vec, Accum sumv) {
  T *view = reinterpret_cast<T *>(vec);
#pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    sumv += (Accum)view[i] * (Accum)view[i];
  }
  return sumv;
}

DEVICE int round_half(half val) {
    return __half2int_rd(__hadd(val, __float2half(0.5f)));
}

/*
 * Given a row index, return the start index of the scale.
*/
HOST_DEVICE int scale_index(int row_id){
  int bottomUpper = (row_id / 8) % 2;
  int group_idx = row_id % 8;
  int group_nums = row_id / 16;
  return (group_nums * 64) + (group_idx * 8) + bottomUpper;
}

/*
 * Given the row numbers, calculate the leading dimension of scales.
 * In unit of half.
*/
#define SCALE_SIZE_A(x) ((x) / 16 * 64 + 64 - (1 - (x % 16) / 8) * (8 - (x % 8)) * 8)

#define mymax(a, b) ((a) > (b) ? (a) : (b))

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
DEVICE Accum local_abs_max(U *vec, Accum maxv) {
  T *view = reinterpret_cast<T *>(vec);
#pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    maxv = mymax((Accum)maxv, (Accum)abs((Accum)view[i]));
  }
  return maxv;
}

template <int bdx, int bdy, int hidden_dim, int group_size>
__global__ void rmsnorm_fp16_i4_kernel(
  half *input,
  half *weight,
  float eps,
  int16_t *reorder_index,
  int8_t *s8out,
  int8_t *s4out,
  half *s8scale,
  half *s4scale,
  int scale_ldm
){
  static_assert(bdx * bdy == 128);
  static_assert(hidden_dim == 4096, "Currently do not support dynamic hidden_dim.");
  static_assert(group_size == 128);
  
  constexpr int elements_per_thread = hidden_dim / (bdx * bdy);
  static_assert(bdx * elements_per_thread == group_size);
  
  // rows are independent, so we can change the pointer at beginning...
  int row_id = blockIdx.x;
  input = input + row_id * hidden_dim;
  s8out = s8out + row_id * group_size;
  s4out = s4out + row_id * (hidden_dim - group_size) / 2; // 2 int4 packed in 1 int8_t

  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ uint8_t raw_smem[];
  half *input_smem = reinterpret_cast<half *>(raw_smem);
  half input_frag[elements_per_thread];
  // Block-level reduction. Using shared memory.
  constexpr int input_smem_bytes = hidden_dim * sizeof(half);
  float *sum_smem = reinterpret_cast<float *>(raw_smem + input_smem_bytes);

  // Load all data and calculate partial sum, using vector load 128bits
  float4 *input_smem_float4 = reinterpret_cast<float4 *>(input_smem);
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  float4 *input_global_float4 = reinterpret_cast<float4 *>(input);
  // Load input_iterations float4. Also the stripe for thread load.
  constexpr int input_iterations = elements_per_thread * sizeof(half) / sizeof(float4);
  int tid = cta.thread_rank();
#pragma unroll
  for(int i = 0;i < input_iterations;++i){
    input_smem_float4[tid * input_iterations + i] =
      input_frag_float4[i] = input_global_float4[tid * input_iterations + i];
  }
  float sumv = 0.f;
#pragma unroll
  for(int i = 0;i < input_iterations;++i){
    sumv = local_sum_p2<half, float4, float>(input_frag_float4 + i, sumv);
  }
  sum_smem[tid] = sumv;
  cta.sync();
  // Block-level reduction. Using shared memory.
  if(tid < 64) {
    sum_smem[tid] = sumv = sumv + sum_smem[tid + 64];
  }
  cta.sync();
  if(tid < 32) {
    sum_smem[tid] = sumv = sumv + sum_smem[tid + 32];
  }
  cta.sync();
  // Within single warp
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  if (tid < 32) {
    for (int s = 16; s > 0; s >>= 1) {
      sumv += tile32.shfl_down(sumv, s);
    }
  }
  // this makes sumv equal to rvariance
  if (tid == 0) {
    sumv = rsqrt(sumv / hidden_dim + eps);
    sum_smem[0] = sumv;
  }
  // broadcast the rvariance of tid 0 to all the threads
  cta.sync();
  sumv = sum_smem[0];

  // reorder input and do normalization
#pragma unroll 4
  for (int i = 0; i < elements_per_thread; ++i) {
    int reordered_idx = (int32_t)reorder_index[tid * elements_per_thread + i];
    half input_value = input_smem[reordered_idx];
    half weight_value = weight[reordered_idx];
    input_frag[i] = __float2half((float)input_value * (float)weight_value * (float)sumv);
  }

  // quantize INT4 and INT8
  int group_id = threadIdx.y;
  // thread local maxv
  float maxv = -1;
  // compute thread local maxv
#pragma unroll 4
  for(int i = 0;i < input_iterations;++i){
    maxv = local_abs_max<half, float4, float>(input_frag_float4 + i, maxv);
  }
  // Reduce within same group
#pragma unroll
  for(int i = bdx / 2; i > 0; i >>= 1){
    float tmp;
    asm volatile(
      "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : 
      "=f"(tmp) : "f"(maxv), "r"(i)
    );
    maxv = mymax(maxv, tmp);
  }

  int replicated_row_id = scale_index(row_id);
  if(group_id == bdy - 1){
    // the last group, INT8 quantize
    maxv /= 127;
    if(threadIdx.x == 0){
#pragma unroll
      for(int j = 0; j < 4; ++j){
        s8scale[replicated_row_id + 2 * j] = (half)maxv;
      }
    }
  }else{
    // INT4 quantize
    maxv /= 7;
    if(threadIdx.x == 0){
#pragma unroll
      for(int j = 0; j < 4; ++j){
        s4scale[group_id * scale_ldm + replicated_row_id + 2 * j] = (half)maxv;
      }
    }
  }

  // view input frag as int8 or pack int4 result
  int8_t *input_frag_int8 = reinterpret_cast<int8_t *>(input_frag);
  PackInt4 *input_frag_pack_int4 = reinterpret_cast<PackInt4 *>(input_frag);
  // reverse for replacing devision by multiplication
  float r_scale = 1 / maxv;
  int lower_bound = (group_id == bdy - 1) ? -128 : -8;
  int upper_bound = (group_id == bdy - 1) ? 127 : 7;
  for (int j = 0; j < elements_per_thread; j += 2) {
    int8_t result0, result1;
    result0 = (int8_t)clamp(round((float)input_frag[j] * r_scale),
                                  lower_bound, upper_bound);
    result1 = (int8_t)clamp(round((float)input_frag[j + 1] * r_scale),
                                  lower_bound, upper_bound);
    if (group_id == bdy - 1) {
      // the last group, INT8 quantize
      input_frag_int8[j] = result0;
      input_frag_int8[j + 1] = result1;
    } else {
      // INT4 quantize
      input_frag_pack_int4[j / 2].low = result0;
      input_frag_pack_int4[j / 2].high = result1;
    }
  }
  // finally, store out
  if(group_id == bdy - 1){
    // the last group, INT8 quantize
    constexpr int output_iterations = input_iterations / (sizeof(half) / sizeof(uint8_t));
    int4 *res = reinterpret_cast<int4 *>(input_frag);
    int4 *s8out_int4 = reinterpret_cast<int4 *>(s8out) + threadIdx.x * output_iterations;
#pragma unroll
    for(int j = 0;j < output_iterations;++j){
      s8out_int4[j] = res[j];
    }
  }else{
    // INT4 quantize
    static_assert(elements_per_thread / 2 == sizeof(int4));
    constexpr int output_iterations = input_iterations / (sizeof(half) * 2 / sizeof(uint8_t));
    int4 *res = reinterpret_cast<int4 *>(input_frag);
    int4 *s4out_int4 = reinterpret_cast<int4 *>(s4out) + threadIdx.x * output_iterations;
#pragma unroll
    for(int j = 0;j < output_iterations;++j){
      s4out_int4[group_id * bdx + j] = res[j];
    }
  }
}

/*!
 * \brief RMS Layer Norm for Llama model. Fused with reorder and quantization.
 * \brief Current only support 128 group size.
 * \tparam group_size Quantization group size
 * \tparam hidden_dim Hidden dimension
 * \param hidden_states Input hidden_states. [seq_len, hidden_dim] row-major
 * \param weight Layer norm weight matrix. [hidden_dim] row-major
 * \param eps Epsilon for layer norm. Avoid div by zero.
 * \param seq_len Sequence length of hidden_states.
 * \param reorder_index Reorder index for hidden_states. [hidden_dim] row-major. Output will be torch.select_index(, dim=1, index=reorder_index)
 * \param o_outliers Quantized INT8 output. [seq_len, group_size] row-major.
 * \param o_norms Quantized INT4 output. [seq_len, hidden_dim - group_size] row-major.
 * \param outlier_scales Quantization scales for INT8 output. Specific layout. See scale_index() for details. Column-major.
 * \param norm_scales Quantization scales for INT4 output. Specific layout. See scale_index() for details. Column-major.
*/
template<int group_size, int hidden_dim>
void run_rmsnorm_fp16_i4(
  half *hidden_states,
  half *weight,
  float eps,
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
){
  static_assert(group_size == 128 && hidden_dim == 4096, "Current only support 128x4096.");
  constexpr int bdx = 4, bdy = 32;
  dim3 grids(seq_len);
  dim3 blocks(bdx, bdy);
  // float for block reduce
  size_t smem_size = hidden_dim * sizeof(half) + bdx * bdy * sizeof(float);

  rmsnorm_fp16_i4_kernel<bdx, bdy, hidden_dim, group_size><<<grids, blocks, smem_size, nullptr>>>(
    (half *)hidden_states,
    (half *)weight,
    eps,
    (int16_t *)reorder_index,
    (int8_t *)o_outliers,
    (int8_t *)o_norms,
    (half *)outlier_scales,
    (half *)norm_scales,
    SCALE_SIZE_A(seq_len)
  );
}

template
void run_rmsnorm_fp16_i4<128, 4096>(
  half *hidden_states,
  half *weight,
  float eps,
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
);
