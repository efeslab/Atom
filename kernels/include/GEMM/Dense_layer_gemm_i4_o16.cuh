#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <random>

// Accumulator: 128 * 128 * sizeof(int32_t) = 64KB
// Block A + B: 128 * 128 * sizeof(int8_t) * 0.5 * 2 = 16KB
// Seems that loading to registers is bottleneck. Since
// with same data size, we have more computation in one block.
// But has less computation in one mma(8x8x32). Maybe we 2-stage
// pipeline in (gmem, smem) and 2-stage pipeline in (smem, reg)
// is best.
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 128

#define BLOCK_WARPS 8
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 2

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 4

#define M 16
#define N 8
#define K 64

#define STAGE 4

// GPU configuration.
#define WARP_SIZE 32

// Quantization configuration
#define GROUP_SIZE 128
// Note: Packed into half2 and copy 4 times -> (M_GLOBAL / 2) * 4
#define SCALE_PACKING_A(x) ((x) * 2)
#define SCALE_PACKING_B(x) ((x) / 2)

// This is only for calculating A's scales number of half2 in unit test
// Calculated by the layout of A Scale matrix
#define SCALE_SIZE_A(x) ((x) / 16 * 32 + 32 - (1 - (x % 16) / 8) * (8 - (x % 8)) * 4)

// 16 Bytes = 128 bits = 32 * sizeof(u4) -> actually per row
// Chunk means per row loading
typedef int4 copy_t;
#define CHUNK_LOAD_BYTES (BLOCK_K * sizeof(int8_t) / 2)
#define CHUNK_LOAD_LANES_PER (CHUNK_LOAD_BYTES / sizeof(copy_t))
#define CHUNK_LOAD_PER_WARP (WARP_SIZE / CHUNK_LOAD_LANES_PER)

#define E2S(x) ((x) >> 1)

// Mixed-precision part
#define KEEPER 128
#define KEEPER_M 16
#define KEEPER_N 8
#define KEEPER_K 32
// Keep aligned with s4 data usage. Use equal size mma 16x8x32 (Only K-dimension shinks 1/2)
// so that we can reuse the data movement function.
// BLOCK_K of s4 -> 128; BLOCK_K of INT8 -> 64
#define KEEPER_BLOCK_K (BLOCK_K / 2 / sizeof(int8_t))

// Load BLOCK_M * BLOCK_K elements from global memory to shared memory
// Cooperative loading within same block with all 8 warps (Block-level)
__device__ __forceinline__ void loadASMem(
  uint8_t *smem,       // Start address of loaded space
  const uint8_t *gmem, // Start address of this block address
  const int max_m_dimension,  // M_GLOBAL
  const int gmem_ldm,  // K_GLOBAL
  const int k,         // Current k offset
  bool predGuard       // To resolve the extra access
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // Note: col index is counted at the granularity of copy_t
  // Row is determined by warp idx & lane idx
  // Col is determined by lane idx 
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  // Deal with M tail block: avoid illegal memory access
  // Check each lane's M-loading dimension to determine whether illegal
  predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);

  // Deal with K tail block: avoid illegal memory access
  // Check the k dimension pointer address.
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);
  
  // @!p st.shared.v4.u32 is bottleneck for 20 Tops drop.
#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
    predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);
  }
}

__device__ __forceinline__ void loadBSMem(
  uint8_t *smem,       // Start address of loaded space
  const uint8_t *gmem, // Start address of this block address
  const int gmem_ldm,  // K_GLOBAL
  const int k,         // Current k offset
  bool predGuard       // To resolve the extra access
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // Note: col index is counted at the granularity of copy_t
  // Row is determined by warp idx & lane idx
  // Col is determined by lane idx 
  int gmem_row = warpId * BLOCK_N / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  // Deal with K tail block: avoid illegal memory access
  // Note: B Matrix is always weight matrix. So we don't consider the tail block of N dimension.
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);

#pragma unroll
  for(int i = 0; i < BLOCK_N / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n" 
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
  }
}

// Block-level function
// Note: need to output [M, N], take care of the M tail blocks
__device__ __forceinline__ void storeSMem(
  const half *smem,           // Start address of loaded space
  half *gmem,                 // Start address of this block address
  const int smem_ldm,
  const int max_m_dimension,  // M_GLOBAL
  const int gmem_ldm          // N_GLOBAL
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // 128 * 16 / 8 = 16 * 16
  // One warp for two chunk
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / (CHUNK_LOAD_LANES_PER * 4);
  int gmem_col = laneId % (CHUNK_LOAD_LANES_PER * 4);
  int smem_row = gmem_row;
  int smem_col = gmem_col;

#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / (CHUNK_LOAD_PER_WARP / 4); ++i){
    if(gmem_row + blockIdx.y * BLOCK_M < max_m_dimension){
      *((copy_t*)(&gmem[gmem_row * gmem_ldm]) + gmem_col) =
        *((copy_t*)(smem + smem_row * smem_ldm) + smem_col);
        
      gmem_row += (CHUNK_LOAD_PER_WARP / 4);
      smem_row += (CHUNK_LOAD_PER_WARP / 4);
    }
  }
}

// Warp-level function
// Input address is warp-level specific
__device__ __forceinline__ void loadAFrag(
  int32_t *a_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
  // Since each ldmatrix can load 4x8x8, we want to reduce the instruction number.
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; i += 1){
    int smem_row = i * M + tid % 16; // 16 x 64
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(a_frag[i * 4]), "=r"(a_frag[i * 4 + 1]), "=r"(a_frag[i * 4 + 2]), "=r"(a_frag[i * 4 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );
  }
}

__device__ __forceinline__ void loadBFrag(
  int32_t *b_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
#pragma unroll
  for(int i = 0;i < WARP_ROW_TILES; i += 2){
    int smem_row = i * N + tid % 16;
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(b_frag[i * 2 + 0]), "=r"(b_frag[i * 2 + 2]), "=r"(b_frag[i * 2 + 1]), "=r"(b_frag[i * 2 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );    
  }
}

// Warp-level function
// Input address is warp-level specific: which means we only cares about single warp.
__device__ __forceinline__ void storeAccumulator(
  float *c_frag,  // [col, row, 2]
  half *smem,        // cast int8_t* to int*
  const int smem_ldm
){
  // According to fragment layout
  const int ti = threadIdx.x % 4;
  const int tj = threadIdx.x / 4;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      half *ptr = &smem[i * smem_ldm * M + j * N];
      ptr[tj * smem_ldm + ti * 2 + 0] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]);
      ptr[tj * smem_ldm + ti * 2 + 1] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 0] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 1] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]);
    }
  }
}

template <bool initZero>
__device__ __forceinline__ void mma_calculate(
  int32_t* __restrict__ c_frag,
  int32_t* __restrict__ a_frag,
  int32_t* __restrict__ b_frag
){
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      if constexpr (initZero){
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(0), "r"(0),
            "r"(0), "r"(0)
        );
      }else{
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
        );
      }
    }
  }
}

/*
  Thread Block: 128x128 x64
  Warp: 64x32 x64
  Warp Iteration: 16x8 x32
*/
__device__ __forceinline__ void mma_calculate_keeper(
  int32_t *c_frag,
  int32_t *a_frag,
  int32_t *b_frag
){
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
          "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
        : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
          "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
          "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
          "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
      );
    }
  }
}

/*
  Currently only supports 128 x 128
  Thread block-level function。
  Note: deal with the tail block. No need to consider K dimension since |K| = (K_GLOBAL + GROUP_SIZE - 1) / GROUP_SIZE
  Consider M dimension and N dimension:
    Each lane copy 16 bytes -> half2 x 4
    read_lane_nums + start_m_offset < M_GLOBAL will be ok.
*/
__device__ __forceinline__ void loadScale(
  uint8_t *smem_A_scale,
  uint8_t *smem_B_scale,
  const uint8_t *gmem_A_scale,  // [K_GLOBAL / GROUP_SIZE, M_GLOBAL]
  const uint8_t *gmem_B_scale,  // [K_GLOBAL / GROUP_SIZE, N_GLOBAL]
  const int max_m_dimension,    // M_GLOBAL
  bool predGuard
){
  const int laneId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  // Assert BLOCK_M == BLOCK_N
  constexpr int copySingleSizeA = SCALE_PACKING_A(BLOCK_M) * sizeof(half2) * BLOCK_K / GROUP_SIZE; // 1024
  constexpr int neededLanesA = copySingleSizeA / sizeof(copy_t); // 1024 / 16 = 64
  constexpr int copySingleSizeB = SCALE_PACKING_B(BLOCK_N) * sizeof(half2) * BLOCK_K / GROUP_SIZE; // 256
  constexpr int neededLanesB = copySingleSizeB / sizeof(copy_t); // 256 / 16 = 16

  predGuard = predGuard && (laneId < neededLanesA + neededLanesB);
  bool M_tail_check = ((laneId / 8) * 16 + (laneId % 8) + blockIdx.y * BLOCK_M < max_m_dimension);
  predGuard = predGuard && (M_tail_check || laneId >= neededLanesA);

  if(laneId < neededLanesA){
    copy_t *dst_ptr = (copy_t*) smem_A_scale + laneId;
    copy_t *src_ptr = (copy_t*) gmem_A_scale + laneId;
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n" 
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared(dst_ptr)),
        "l"(src_ptr)
    );
  }else{
    copy_t *dst_ptr = (copy_t*) smem_B_scale + (laneId - neededLanesA);
    copy_t *src_ptr = (copy_t*) gmem_B_scale + (laneId - neededLanesA);
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n" 
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared(dst_ptr)),
        "l"(src_ptr)
    );
  }
}

// Warp-level function
// To load scale from shared memory to register
__device__ __forceinline__ void loadScaleReg(
  int32_t *reg_a,
  int32_t *reg_b,
  const uint8_t *smem_A_scale,
  const uint8_t *smem_B_scale
){
  const int tid = threadIdx.x;
  copy_t *ptr = (copy_t *)smem_A_scale + tid;
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    :  "=r"(reg_a[0]), "=r"(reg_a[1]), "=r"(reg_a[2]), "=r"(reg_a[3])
    :  "l"(__cvta_generic_to_shared(ptr))
  );
  ptr = (copy_t *)smem_B_scale + tid / 8;
  // Suboptimal for redundant loading same scales
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    :  "=r"(reg_b[0]), "=r"(reg_b[1]), "=r"(reg_b[2]), "=r"(reg_b[3])
    :  "l"(__cvta_generic_to_shared(ptr))
  );
}

__device__ __forceinline__ void dequant(
  int32_t *c_frag,
  int32_t *reg_a,
  int32_t *reg_b,
  float *accu
){
  // TODO: half multiplication may have accuracy drop
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
    half2 row_scale = *(half2*)(&reg_a[i]);
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      half2 col_scale = *(half2*)(&reg_b[j]);
      half2 rs_scale = __hmul2(row_scale, col_scale);
      float rs_scale_u = __half2float(rs_scale.x);
      float rs_scale_d = __half2float(rs_scale.y);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 0] += 
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]) *
        rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 1] += 
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]) *
        rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 2] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]) *
        rs_scale_d;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 3] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]) *
        rs_scale_d;
    }
  }
}

__global__ void __launch_bounds__(256) compute_gemm_imma(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  const uint8_t *A_keeper,
  const uint8_t *B_keeper,
  const uint8_t *A_keeper_scale,
  const uint8_t *B_keeper_scale
){
  extern __shared__ uint8_t shmem[];

  const size_t shmem_B_offset = E2S(BLOCK_M * BLOCK_K * sizeof(int8_t));
  const size_t shmem_stage_offset = E2S(BLOCK_K * (BLOCK_M + BLOCK_N) * sizeof(int8_t));
  const size_t shmem_scale_offset = STAGE * shmem_stage_offset;
  const size_t shmem_scale_B_offset = SCALE_PACKING_A(BLOCK_M) * BLOCK_K / GROUP_SIZE * sizeof(half2);
  const size_t shmem_scale_stage_offset = (SCALE_PACKING_A(BLOCK_M) + SCALE_PACKING_B(BLOCK_N)) * BLOCK_K / GROUP_SIZE * sizeof(half2);
  const int A_scale_stride = SCALE_SIZE_A(M_GLOBAL);

  const int bi = blockIdx.x; // N dimension
  const int bj = blockIdx.y; // M dimension
  const int wi = threadIdx.y;
  const int wj = threadIdx.z;

  // m16n8k32 -> 16*32 accumulator
  int32_t c[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0};
  int32_t a[2][WARP_COL_TILES * 4] = {0};
  int32_t b[2][WARP_ROW_TILES * 2] = {0};
  // real accumulator
  float c_fp[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0.0f};
  int32_t a_s[WARP_COL_TILES] = {0};
  int32_t b_s[WARP_ROW_TILES] = {0};

  // Each time (writePtr + 1) % STAGE is consumed, and writePtr is produced.
  size_t writePtr = STAGE - 1;
  // Keep one unused stage for producing
#pragma unroll
  for(int i = 0; i < STAGE - 1;++i){
    loadASMem(
      shmem + i * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + i * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    loadBSMem(
      shmem + shmem_B_offset + i * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + i * BLOCK_K),
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    loadScale(
      shmem + shmem_scale_offset + i * shmem_scale_stage_offset,
      shmem + shmem_scale_offset + shmem_scale_B_offset + i * shmem_scale_stage_offset,
      A_scale + sizeof(half2) * ((i * BLOCK_K) / GROUP_SIZE * A_scale_stride + SCALE_PACKING_A(bj * BLOCK_M)),
      B_scale + sizeof(half2) * SCALE_PACKING_B((i * BLOCK_K) / GROUP_SIZE * N_GLOBAL + bi * BLOCK_N),
      M_GLOBAL,
      true
    );
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
  __syncthreads();

  loadAFrag(
    a[0],
    shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  loadBFrag(
    b[0],
    shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  // Main loop is calculated at the unit of calculation
  // Use predicate reg to avoid unnecessary if instruction
  for(int k = 0; k < K_GLOBAL; k += BLOCK_K){
    loadAFrag(
      a[1],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadBFrag(
      b[1],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadScaleReg(
      a_s,
      b_s,
      shmem + shmem_scale_offset + (writePtr + 1) % STAGE * shmem_scale_stage_offset + sizeof(half2) * SCALE_PACKING_A(wj * WARP_COL_TILES * M),
      shmem + shmem_scale_offset + shmem_scale_B_offset + (writePtr + 1) % STAGE * shmem_scale_stage_offset + sizeof(half2) * SCALE_PACKING_B(wi * WARP_ROW_TILES * N)
    );
    mma_calculate<true>(c, a[0], b[0]);
    // Pipeline load
    bool predGuard = (k + (STAGE - 1) * BLOCK_K) < K_GLOBAL;
    loadASMem(
      shmem + writePtr * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    loadBSMem(
      shmem + shmem_B_offset + writePtr * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    loadScale(
      shmem + shmem_scale_offset + writePtr * shmem_scale_stage_offset,
      shmem + shmem_scale_offset + shmem_scale_B_offset + writePtr * shmem_scale_stage_offset,
      A_scale + sizeof(half2) * ((k + (STAGE - 1) * BLOCK_K) / GROUP_SIZE * A_scale_stride + SCALE_PACKING_A(bj * BLOCK_M)),
      B_scale + sizeof(half2) * SCALE_PACKING_B((k + (STAGE - 1) * BLOCK_K) / GROUP_SIZE * N_GLOBAL + bi * BLOCK_N),
      M_GLOBAL,
      predGuard
    );
    asm volatile("cp.async.commit_group;\n" ::);
    mma_calculate<false>(c, a[1], b[1]);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
    writePtr = (writePtr + 1) % STAGE;
    __syncthreads();
    loadAFrag(
      a[0],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    loadBFrag(
      b[0],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    dequant(
      c,
      a_s,
      b_s,
      c_fp
    );
  }

  writePtr = STAGE - 1;
  // Load all INT8 scales.
  loadScale(
    shmem + shmem_scale_offset ,
    shmem + shmem_scale_offset + shmem_scale_B_offset,
    A_keeper_scale + sizeof(half2) * SCALE_PACKING_A(bj * BLOCK_M), // Do not need SCALE_SIZE_A for BLOCK level constant.
    B_keeper_scale + sizeof(half2) * SCALE_PACKING_B(bi * BLOCK_N),
    M_GLOBAL,
    true
  );

  // Here Assume all load instruction are issued.
#pragma unroll
  for(int i = 0; i < STAGE - 1;++i){
    bool predGuard = (i * KEEPER_BLOCK_K) < KEEPER;
    loadASMem(
      shmem + i * shmem_stage_offset,
      A_keeper + sizeof(int8_t) * (bj * BLOCK_M * KEEPER + i * KEEPER_BLOCK_K),
      M_GLOBAL,
      KEEPER * 2, // * 2 for cancel E2S's effect
      (i * KEEPER_BLOCK_K),
      predGuard
    );
    loadBSMem(
      shmem + shmem_B_offset + i * shmem_stage_offset,
      B_keeper + sizeof(int8_t) * (bi * BLOCK_N * KEEPER + i * KEEPER_BLOCK_K),
      KEEPER * 2, // for cancel E2S
      (i * KEEPER_BLOCK_K),
      predGuard
    );
    // Allow to commit an empty async cp group.
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
  __syncthreads();

  loadAFrag(
    a[0],
    shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  loadBFrag(
    b[0],
    shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );

  // Clean the accumulator for Keeper calculation
  for(int k = 0; k < WARP_COL_TILES * WARP_ROW_TILES * 4; ++k) c[k] = 0;
  // Calculate the mixed-precision part
#pragma unroll
  for(int k = 0; k < KEEPER; k += KEEPER_BLOCK_K){
    loadAFrag(
      a[1],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadBFrag(
      b[1],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    // Do not load: Assume KEEPER can be loaded in prelogue (KEEPER <= 3*64)
    mma_calculate_keeper(c, a[0], b[0]);
    // empty group to make it correct
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
    writePtr = (writePtr + 1) % STAGE;
    __syncthreads();
    mma_calculate_keeper(c, a[1], b[1]);
    loadAFrag(
      a[0],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    loadBFrag(
      b[0],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
  }
  
  // Load scale into register for once
  loadScaleReg(
    a_s,
    b_s,
    shmem + shmem_scale_offset + sizeof(half2) * SCALE_PACKING_A(wj * WARP_COL_TILES * M),
    shmem + shmem_scale_offset + shmem_scale_B_offset + sizeof(half2) * SCALE_PACKING_B(wi * WARP_ROW_TILES * N)
  );

  dequant(
    c,
    a_s,
    b_s,
    c_fp
  );

  // Offload accumulator from registers to shared memory
  // Extra stage is for avoiding random access to shared memory
  storeAccumulator(
    c_fp,
    (half *)shmem + wj * WARP_COL_TILES * M * BLOCK_N + wi * WARP_ROW_TILES * N,
    BLOCK_N
  );
  __syncthreads();

  // Write back to global memory
  storeSMem(
    (half *)shmem,
    (half *)D + bj * BLOCK_M * N_GLOBAL + bi * BLOCK_N,
    BLOCK_N,
    M_GLOBAL,
    N_GLOBAL
  );
}

/*!
 * \brief Dense GEMM kernel for mixed-precision. A,B for INT4 tensor core; A_keeper, B_keeper for INT8 tensor core.
 * \brief Assume keeper size = 128 && quantization group size = 128
 * \param A INT4 matrix in global memory. Packed in uint8_t. [M, K-128] row-major
 * \param B INT4 matrix in global memory. Packed in uint8_t. [K-128, N] column-major
 * \param A_scale Scale for A. nv_half type while input as uint8_t pointer. [M, (K-128) // 128] with specific layout
 * \param B_scale Scale for B. nv_half type while input as uint8_t pointer. [(K-128) // 128, N] with specific layout
 * \param A_keeper INT8 matrix in global memory. [M, 128] row-major
 * \param B_keeper INT8 matrix in global memory. [128, N] column-major
 * \param A_keeper_scale Scale for A_keeper. nv_half type while input as uint8_t pointer. [M, 1] with specific layout
 * \param B_keeper_scale Scale for B_keeper. nv_half type while input as uint8_t pointer. [1, N] with specific layout
 * \param D Output matrix in global memory. [M, N] row-major. Type \in [half, fp8, float]
 * \param M_GLOBAL Number of rows of matrix op(A).
 * \param K_GLOBAL hidden dimension \in [4096, 11008]
 * \param N_GLOBAL Number of columns of matrix op(B). Should be 4096 / 11008
*/
void DenseLayerGEMM_i4_o16(
  const uint8_t *A,
  const uint8_t *B,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  const uint8_t *A_keeper,
  const uint8_t *B_keeper,
  const uint8_t *A_keeper_scale,
  const uint8_t *B_keeper_scale,
  half *D,
  const size_t M_GLOBAL,
  const size_t N_GLOBAL,
  const size_t K_GLOBAL
){
  dim3 gridDim(
    (N_GLOBAL + BLOCK_N - 1) / BLOCK_N,
    (M_GLOBAL + BLOCK_M - 1) / BLOCK_M
  );
  dim3 blockDim(
    WARP_SIZE,
    BLOCK_ROW_WARPS,
    BLOCK_COL_WARPS
  );

  constexpr size_t shmem_size1 = sizeof(uint8_t) * BLOCK_K * (BLOCK_M + BLOCK_N) / 2 * STAGE +
      sizeof(half2) * (SCALE_PACKING_A(BLOCK_M) + SCALE_PACKING_B(BLOCK_N)) * STAGE;
  constexpr size_t shmem_size2 = BLOCK_M * BLOCK_N * sizeof(half);
  constexpr size_t SHMEM_SZ = shmem_size1 > shmem_size2 ? shmem_size1 : shmem_size2;

  cudaFuncSetAttribute(
    compute_gemm_imma,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SZ
  );
  
  compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ>>>(
  A, B, D,
  M_GLOBAL, N_GLOBAL, K_GLOBAL,
  A_scale, B_scale,
  A_keeper, B_keeper,
  A_keeper_scale, B_keeper_scale);
}
