#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

#define UNROLL #pragma unroll
#define NO_UNROLL #pragma unroll 1

namespace cg = cooperative_groups;

struct PackInt4 {
  int8_t low : 4;
  int8_t high : 4;
};

HOST_DEVICE int cdiv(int a, int b) { return (a + b - 1) / b; }

HOST_DEVICE int clamp(int x, int a, int b) { return max(a, min(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }

// SiLu activation function, Ref:https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
HOST_DEVICE float silu(float x) { return x / (1.0f + expf(-x)); }

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
HOST_DEVICE Accum local_sum_p2(U *vec, Accum sumv) {
  T *view = reinterpret_cast<T *>(vec);
  UNROLL 4
  for (int i = 0; i < Size; ++i) {
    sumv += (Accum)view[i] * (Accum)view[i];
  }
  return sumv;
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

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
DEVICE float local_abs_max(U *vec, float maxv) {
  T *view = reinterpret_cast<T *>(vec);
  UNROLL 4
  for (int i = 0; i < Size; ++i) {
    maxv = mymax((float)maxv, (float)abs((float)view[i]));
  }
  return maxv;
}

template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void activate_fp16_i4_kernel(
  half *A,
  half *B,
  int8_t *s8out,
  int8_t *s4out,
  half *s8scale,
  half *s4scale,
  int scale_ldm
){
  static_assert(GROUP_SIZE == 128, "Current only support 128 group size.");
  static_assert(bdx == 32, "Current only supports one warp deal with one group.");
  constexpr int elements_per_thread = GROUP_SIZE / bdx;

  cg::thread_block cta = cg::this_thread_block();

  // Local memory stores input value;
  half input_A[elements_per_thread];
  half input_B[elements_per_thread];
  float input_float[elements_per_thread];

  // Row and group are independent
  int group_id = blockIdx.x;
  int row_id = blockIdx.y;
  A = A + row_id * HIDDEN_DIM + group_id * GROUP_SIZE;
  B = B + row_id * HIDDEN_DIM + group_id * GROUP_SIZE;
  s8out = s8out + row_id * GROUP_SIZE;
  // Pack two int4 into one int8
  s4out = s4out + row_id * (HIDDEN_DIM - GROUP_SIZE) / 2 + group_id * GROUP_SIZE / 2;

  // Below are operations within single warp
  int tid = threadIdx.x;
  *(reinterpret_cast<float2*>(input_A)) = *(reinterpret_cast<float2*>(A) + tid);
  *(reinterpret_cast<float2*>(input_B)) = *(reinterpret_cast<float2*>(B) + tid);

  // Calculate the activation
  UNROLL
  for(int i = 0; i < elements_per_thread; ++i){
    input_float[i] = silu((float)input_A[i]) * (float)input_B[i];
  }
  // Reduce to get max
  // Each thread get partial max. Then warp reduces to get the max of this warp.
  float maxv = -65536.f;
  maxv = local_abs_max<float, float4>(reinterpret_cast<float4*>(input_float), maxv);
  cta.sync();
  // Warp reduce
  cg::thread_block_tile<32> warpTile = cg::tiled_partition<32>(cta);
  UNROLL
  for(int offset = 16; offset > 0; offset /= 2){
    float tmpReduce = warpTile.shfl_down(maxv, offset);
    maxv = mymax(maxv, tmpReduce);
  }
  // Broadcast to all threads within this group
  maxv = warpTile.shfl(maxv, 0);

  // Calculate scales
  // Specific layout
  int replicated_row_id = scale_index(row_id);
  if (group_id == gridDim.x - 1) {
    // the last group, INT8 quantize
    maxv /= 127;
    // Only the first thread of each group writes out
    if(tid == 0){
      UNROLL
      for(int j = 0; j < 4; ++j){
        s8scale[replicated_row_id + 2 * j] = (half) maxv;
      }
    }
  } else {
    // INT4 quantize
    maxv /= 7;
    if(tid == 0){
      UNROLL
      for(int j = 0; j < 4; ++j){
        s4scale[group_id * scale_ldm + replicated_row_id + 2 * j] = (half)maxv;
      }
    }
  }

  // Use r_scale to reduce devision
  float r_scale = 1.f / maxv;
  // Quantize each thread's value
  int lower_bound = (group_id == gridDim.x - 1) ? -128 : -8;
  int upper_bound = (group_id == gridDim.x - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  // Reuse the registers
  int8_t* input_frag_int8 = reinterpret_cast<int8_t*>(input_A);
  PackInt4* input_frag_int4 = reinterpret_cast<PackInt4*>(input_A);
  for(int i = 0; i < elements_per_thread; i += 2){
    int8_t result_0, result_1;
    result_0 = (int8_t)clamp(round((float)input_float[i] * r_scale), lower_bound, upper_bound);
    result_1 = (int8_t)clamp(round((float)input_float[i + 1] * r_scale), lower_bound, upper_bound);
    if(group_id == gridDim.x - 1){
      input_frag_int8[i] = result_0;
      input_frag_int8[i + 1] = result_1;
    } else {
      input_frag_int4[i / 2].low = result_0;
      input_frag_int4[i / 2].high = result_1;
    }
  }
  // Store frag out to global memory
  // each threads store 4 elements.
  // INT4: 16bits
  // INT8: 32bits
  if(group_id == gridDim.x - 1){
    // Store int8_t quantized result
    int32_t* s8out_store = reinterpret_cast<int32_t*>(s8out);
    s8out_store[tid] = *(reinterpret_cast<int32_t*>(input_A));
  }else{
    // Store int4_t quantized result
    int16_t* s4out_store = reinterpret_cast<int16_t*>(s4out);
    s4out_store[tid] = *(reinterpret_cast<int16_t*>(input_A));
  }
}

/*!
 * \brief Activation in MLP fused with quantization. quant(SiLU(A) * B)
 * \brief Current only support 128 group size.
 * \tparam group_size Quantization group size
 * \tparam hidden_dim Hidden dimension. Must be multiple of group_size.
 * \param A Input tensor. [seq_len, hidden_dim] row-major.
 * \param B Weight tensor. [seq_len, hidden_dim] row-major.
 * \param o_outliers Quantized INT8 output. [seq_len, group_size] row-major.
 * \param o_norms Quantized INT4 output. [seq_len, hidden_dim - group_size] row-major.
 * \param outlier_scales Quantization scales for INT8 output. Specific layout. See scale_index() for details. Column-major.
 * \param norm_scales Quantization scales for INT4 output. Specific layout. See scale_index() for details. Column-major.
*/
template<int group_size, int hidden_dim>
void run_activate_fp16_i4(
  half *A,
  half *B,
  int seq_len,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
){
  static_assert(group_size == 128, "Current only support 128x4096.");
  static_assert(hidden_dim % group_size == 0, "hidden_dim must be multiple of group_size.");
  dim3 grids(hidden_dim / group_size, seq_len);
  dim3 blocks(32);

  activate_fp16_i4_kernel<32, group_size, hidden_dim><<<grids, blocks>>>(
    (half *)A,
    (half *)B,
    (int8_t *)o_outliers,
    (int8_t *)o_norms,
    (half *)outlier_scales,
    (half *)norm_scales,
    SCALE_SIZE_A(seq_len)
  );
}