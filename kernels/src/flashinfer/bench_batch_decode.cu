#include <thrust/device_vector.h>

#include <cstdint>
#include <flashinfer/decode.cuh>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "utils.h"

using utils::vec_bytes;
using namespace flashinfer::quant;

template <typename QType, typename KVType, typename OType, int FoldFactor>
void bench_flashinfer_batch_decode(nvbench::state& state) {
  // Use uint8_t to allocate memory bytes
  using fakeT = uint8_t;

  constexpr size_t num_heads = 32;
  constexpr size_t head_dim = 128;
  constexpr size_t num_layers = 3;
  constexpr size_t layer_idx = 1;
  constexpr auto rotary_mode = flashinfer::RotaryMode::kNone;
  size_t seqlen = state.get_int64("seqlen");
  size_t batch_size = state.get_int64("batch_size");
  size_t page_size = state.get_int64("page_size");

  // KV cache:
  auto pages_per_seq = (seqlen + page_size - 1) / page_size;
  auto num_pages = pages_per_seq * batch_size;
  std::vector<int32_t> kv_indptr_host{0};
  std::vector<int32_t> kv_indicies_host;
  std::vector<int32_t> kv_last_page_offset_host;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t p = 0; p < pages_per_seq; ++p) {
      kv_indicies_host.push_back(i * pages_per_seq + p);
    }
    kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
    kv_last_page_offset_host.push_back((seqlen - 1) % page_size + 1);
  }
  thrust::device_vector<fakeT> kv_data(num_pages * num_layers * 2 * num_heads * page_size *
                                       head_dim / 2);
  thrust::device_vector<half2> kv_param(num_pages * num_layers * 2 * num_heads * page_size);
  thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
  thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
  thrust::device_vector<int32_t> kv_last_page_offset(kv_last_page_offset_host);

  flashinfer::paged_kv_t<KVType, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      reinterpret_cast<KVType*>(thrust::raw_pointer_cast(kv_data.data())),
      thrust::raw_pointer_cast(kv_param.data()), thrust::raw_pointer_cast(kv_indptr.data()),
      thrust::raw_pointer_cast(kv_indices.data()),
      thrust::raw_pointer_cast(kv_last_page_offset.data()));

  // Allocate input data:
  thrust::device_vector<QType> q(batch_size * num_heads * head_dim);
  thrust::device_vector<OType> o(batch_size * num_heads * head_dim);

  state.add_global_memory_reads<uint8_t>(
      vec_bytes(q) + (num_pages * 2 * num_heads * page_size * head_dim) / 2 + vec_bytes(kv_indptr) +
          vec_bytes(kv_indices) + vec_bytes(kv_last_page_offset) + vec_bytes(kv_param),
      "Read");
  state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");

  state.exec([&](nvbench::launch&) {
    cudaError_t status =
        flashinfer::BatchDecodeWithPagedKVCache<QType, KVType, OType, int32_t, FoldFactor>(
            thrust::raw_pointer_cast(q.data()), paged_kv, thrust::raw_pointer_cast(o.data()),
            rotary_mode);
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_DECODE(QType, KVType, OType, FoldFactor)                        \
  auto bench_flashinfer_batch_decode_##QType##_##KVType##_##OType##_##FoldFactor##_ =          \
      bench_flashinfer_batch_decode<QType, KVType, OType, FoldFactor>;                         \
  NVBENCH_BENCH(bench_flashinfer_batch_decode_##QType##_##KVType##_##OType##_##FoldFactor##_)  \
      .set_name("bench_flashinfer_batch_decode_" STR(QType) "_" STR(KVType) "_" STR(           \
          OType) "_" STR(FoldFactor))                                                          \
      .add_int64_axis("seqlen", {1024}) \
      .add_int64_axis("batch_size",                                                            \
                      {8,16,32,64,128})           \
      .add_int64_axis("page_size", {8,16,32})

// BENCH_FLASHINFER_BATCH_DECODE(half, __precision__s4, half, 1);
BENCH_FLASHINFER_BATCH_DECODE(half, __precision__s4, half, 2);
// BENCH_FLASHINFER_BATCH_DECODE(half, __precision__s4, half, 4);
