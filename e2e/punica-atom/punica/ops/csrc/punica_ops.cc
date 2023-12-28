#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "Activate/Activate.h"
#include "GEMM/DenseLayerGEMM_i4.h"
#include "GEMM/DenseLayerGEMM_i4_o4.h"
#include "Norm/RMSNorm.h"
#include "Reorder/Reorder.h"
#include "flashinfer_adapter/flashinfer_config.h"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(scalar_type, ...) \
  [&]() -> bool {                          \
    switch (scalar_type) {                 \
      __VA_ARGS__                          \
      default:                             \
        return false;                      \
    }                                      \
  }()

#define _DISPATCH_CASE(enum_type, c_type_, ...) \
  case enum_type: {                             \
    using c_type = c_type_;                     \
    return __VA_ARGS__();                       \
  }

#define _DISPATCH_CASES(...)                                 \
  _DISPATCH_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_CASES(__VA_ARGS__))

void activate_fp16_i4(torch::Tensor A, torch::Tensor B, int seq_len,
                      torch::Tensor o_outliers, torch::Tensor o_norms,
                      torch::Tensor outlier_scales, torch::Tensor norm_scales) {
  run_activate_fp16_i4<128, 11008>(
      (nv_half *)A.data_ptr(), (nv_half *)B.data_ptr(), seq_len,
      (int8_t *)o_outliers.data_ptr(), (int8_t *)o_norms.data_ptr(),
      (nv_half *)outlier_scales.data_ptr(), (nv_half *)norm_scales.data_ptr());
}

void batch_decode_i4(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_i4<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void dense_layer_gemm_i4_o4(torch::Tensor a, torch::Tensor b,
                            torch::Tensor a_scale, torch::Tensor b_scale,
                            torch::Tensor a_keeper, torch::Tensor b_keeper,
                            torch::Tensor a_keeper_scale,
                            torch::Tensor b_keeper_scale, torch::Tensor d,
                            torch::Tensor d_scale) {
  DenseLayerGEMM_i4_o4(
      (const uint8_t *)a.data_ptr(), (const uint8_t *)b.data_ptr(),
      (const uint8_t *)a_scale.data_ptr(), (const uint8_t *)b_scale.data_ptr(),
      (const uint8_t *)a_keeper.data_ptr(),
      (const uint8_t *)b_keeper.data_ptr(),
      (const uint8_t *)a_keeper_scale.data_ptr(),
      (const uint8_t *)b_keeper_scale.data_ptr(), (uint8_t *)d.data_ptr(),
      a.size(0), b.size(0), a.size(1) * 2 + a_keeper.size(1),
      (nv_half2 *)d_scale.data_ptr());
}

void dense_layer_gemm_i4_fp16(torch::Tensor a, torch::Tensor b,
                              torch::Tensor a_scale, torch::Tensor b_scale,
                              torch::Tensor a_keeper, torch::Tensor b_keeper,
                              torch::Tensor a_keeper_scale,
                              torch::Tensor b_keeper_scale, torch::Tensor d) {
  DenseLayerGEMM_i4<nv_half>(
      (const uint8_t *)a.data_ptr(), (const uint8_t *)b.data_ptr(),
      (const uint8_t *)a_scale.data_ptr(), (const uint8_t *)b_scale.data_ptr(),
      (const uint8_t *)a_keeper.data_ptr(),
      (const uint8_t *)b_keeper.data_ptr(),
      (const uint8_t *)a_keeper_scale.data_ptr(),
      (const uint8_t *)b_keeper_scale.data_ptr(), (nv_half *)d.data_ptr(),
      a.size(0), b.size(0), a.size(1) * 2 + a_keeper.size(1));
}

void rmsnorm_fp16_i4(torch::Tensor hidden_states, torch::Tensor weight,
                     float eps, torch::Tensor reorder_index,
                     torch::Tensor o_outliers, torch::Tensor o_norms,
                     torch::Tensor outlier_scales, torch::Tensor norm_scales) {
  run_rmsnorm_fp16_i4<128, 4096>(
      (nv_half *)hidden_states.data_ptr(), (nv_half *)weight.data_ptr(), eps,
      hidden_states.size(0), (int16_t *)reorder_index.data_ptr(),
      (int8_t *)o_outliers.data_ptr(), (int8_t *)o_norms.data_ptr(),
      (nv_half *)outlier_scales.data_ptr(), (nv_half *)norm_scales.data_ptr());
}

void reorder_fp16_i4(torch::Tensor hidden_states, torch::Tensor reorder_index,
                     torch::Tensor o_outliers, torch::Tensor o_norms,
                     torch::Tensor outlier_scales, torch::Tensor norm_scales) {
  run_reorder_fp16_i4<128, 4096>(
      (nv_half *)hidden_states.data_ptr(), hidden_states.size(0),
      (int16_t *)reorder_index.data_ptr(), (int8_t *)o_outliers.data_ptr(),
      (int8_t *)o_norms.data_ptr(), (nv_half *)outlier_scales.data_ptr(),
      (nv_half *)norm_scales.data_ptr());
}

}  // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("activate_fp16_i4", &activate_fp16_i4, "");
  m.def("batch_decode_i4", &batch_decode_i4, "");
  m.def("init_kv_i4", &init_kv_i4, "");
  m.def("append_kv_i4", &append_kv_i4, "");
  m.def("dense_layer_gemm_i4_fp16", &dense_layer_gemm_i4_fp16, "");
  m.def("dense_layer_gemm_i4_o4", &dense_layer_gemm_i4_o4, "");
  m.def("rmsnorm_fp16_i4", &rmsnorm_fp16_i4, "");
  m.def("reorder_fp16_i4", &reorder_fp16_i4, "");
}
