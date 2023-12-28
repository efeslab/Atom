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

template <int bdx, int bdy, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void reorder_fp16_i4_kernel(
  half *input,
  int16_t *reorder_index,
  int8_t *s8out,
  int8_t *s4out,
  half *s8scale,
  half *s4scale,
  int scale_ldm
){
  static_assert(GROUP_SIZE == 128 && HIDDEN_DIM == 4096, "Current only support 128x4096.");
  static_assert(bdx * bdy == 128, "Current 128 threads per block.");
  static_assert(bdy == HIDDEN_DIM / GROUP_SIZE, "Current only support 4096/128.");
  constexpr int elements_per_thread = GROUP_SIZE / bdx;

  cg::thread_block cta = cg::this_thread_block();

  // One block solves one row of hidden states.
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(half)];
  half *input_smem = reinterpret_cast<half*>(smem);

  // Local memory stores the reordered hidden states.
  half input_frag[elements_per_thread];

  // Row are independent
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  s8out = s8out + row_id * GROUP_SIZE;
  s4out = s4out + row_id * (HIDDEN_DIM - GROUP_SIZE) / 2; // Pack two int4 into one int8

  // Coalesced access global memory
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = tx + ty * bdx;
  constexpr int bytes_per_iter = bdx * bdy * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(half) / bytes_per_iter;

  UNROLL
  for(int i = 0;i < iters;++i){
    // Each thread loads 16 bytes
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  cta.sync();
  // Reorder
  UNROLL 4
  for(int i = 0;i < elements_per_thread;++i){
    int offset = ty * GROUP_SIZE + tx * elements_per_thread + i;
    input_frag[i] = input_smem[reorder_index[offset]];
  }
  // Reduce to get max
  // Each ty should get its max value
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(half) / sizeof(float4);
  float maxv = -65536.f;

  UNROLL
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<half, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();
  // Sub-warp reduce, using shfl.bfly
  #pragma unroll
  for(int i = bdx / 2; i > 0; i >>= 1){
    float tmp;
    asm volatile(
      "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : 
      "=f"(tmp) : "f"(maxv), "r"(i)
    );
    maxv = mymax(maxv, tmp);
  }
  // Calculate scales
  // Specific layout
  int replicated_row_id = scale_index(row_id);
  if (ty == bdy - 1) {
    // the last group, INT8 quantize
    maxv /= 127;
    // Only the first thread of each group writes out
    if(tx == 0){
      UNROLL
      for(int j = 0; j < 4; ++j){
        s8scale[replicated_row_id + 2 * j] = (half) maxv;
      }
    }
  } else {
    // INT4 quantize
    maxv /= 7;
    if(tx == 0){
      UNROLL
      for(int j = 0; j < 4; ++j){
        s4scale[ty * scale_ldm + replicated_row_id + 2 * j] = (half)maxv;
      }
    }
  }

  // Use reverse scale to replace devision by multiplication
  float r_scale = 1 / maxv;

  // Quantize each thread's value
  int lower_bound = (ty == bdy - 1) ? -128 : -8;
  int upper_bound = (ty == bdy - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  int8_t* input_frag_int8 = reinterpret_cast<int8_t*>(input_frag);
  PackInt4* input_frag_int4 = reinterpret_cast<PackInt4*>(input_frag);
  for(int i = 0; i < elements_per_thread; i += 2){
    int8_t result_0, result_1;
    result_0 = (int8_t)clamp(round((float)input_frag[i] * r_scale), lower_bound, upper_bound);
    result_1 = (int8_t)clamp(round((float)input_frag[i + 1] * r_scale), lower_bound, upper_bound);
    if(ty == bdy - 1){
      input_frag_int8[i] = result_0;
      input_frag_int8[i + 1] = result_1;
    } else {
      input_frag_int4[i / 2].low = result_0;
      input_frag_int4[i / 2].high = result_1;
    }
  }
  // Store frag out to global memory
  if(ty == bdy - 1){
    // Store int8_t quantized result
    float4* s8out_float4 = reinterpret_cast<float4*>(s8out);
    s8out_float4[tx * 2 + 0] = input_frag_float4[0];
    s8out_float4[tx * 2 + 1] = input_frag_float4[1];
  }else{
    // Store int4_t quantized result
    float4* s4out_float4 = reinterpret_cast<float4*>(s4out);
    s4out_float4[tx + ty * bdx] = input_frag_float4[0];
  }
}

/*!
 * \brief RMS Layer Norm for Llama model. Fused with reorder and quantization.
 * \brief Current only support 128 group size.
 * \tparam group_size Quantization group size
 * \tparam hidden_dim Hidden dimension
 * \param hidden_states Input hidden_states. [seq_len, hidden_dim] row-major
 * \param seq_len Sequence length of hidden_states.
 * \param reorder_index Reorder index for hidden_states. [hidden_dim] row-major. Output will be torch.select_index(, dim=1, index=reorder_index)
 * \param o_outliers Quantized INT8 output. [seq_len, group_size] row-major.
 * \param o_norms Quantized INT4 output. [seq_len, hidden_dim - group_size] row-major.
 * \param outlier_scales Quantization scales for INT8 output. Specific layout. See scale_index() for details. Column-major.
 * \param norm_scales Quantization scales for INT4 output. Specific layout. See scale_index() for details. Column-major.
*/
template<int group_size, int hidden_dim>
void run_reorder_fp16_i4(
  half *hidden_states,
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
){
  static_assert(group_size == 128 && hidden_dim == 4096, "Current only support 128x4096.");
  dim3 grids(seq_len);
  dim3 blocks(4, 32);

  reorder_fp16_i4_kernel<4, 32, group_size, hidden_dim><<<grids, blocks>>>(
    (half *)hidden_states,
    (int16_t *)reorder_index,
    (int8_t *)o_outliers,
    (int8_t *)o_norms,
    (half *)outlier_scales,
    (half *)norm_scales,
    SCALE_SIZE_A(seq_len)
  );
}

template
void run_reorder_fp16_i4<128, 4096>(
  half *hidden_states,
  int seq_len,
  int16_t *reorder_index,
  int8_t *o_outliers,
  int8_t *o_norms,
  half *outlier_scales,
  half *norm_scales
);
