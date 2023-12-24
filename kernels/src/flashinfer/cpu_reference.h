#pragma once

#include <flashinfer.cuh>

#include "utils.h"

namespace cpu_reference {

using namespace flashinfer;

template <typename T>
inline std::vector<float> apply_llama_rope(const T* input, size_t D, size_t offset,
                                           float rope_scale, float rope_theta) {
  std::vector<float> rst(D);
  std::vector<float> permuted_input(D);
  for (size_t k = 0; k < D; ++k) {
    permuted_input[k] = (k < D / 2) ? -float(input[k + D / 2]) : float(input[k - D / 2]);
  }

  for (size_t k = 0; k < D; ++k) {
    float inv_freq =
        (offset / rope_scale) / (std::pow(rope_theta, float(2 * (k % (D / 2))) / float(D)));
    float cos = std::cos(inv_freq);
    float sin = std::sin(inv_freq);
    rst[k] = cos * float(input[k]) + sin * permuted_input[k];
  }
  return std::move(rst);
}

template <typename dtype_in, typename dtype_out>
std::vector<dtype_out> single_mha(const std::vector<dtype_in>& q, const std::vector<dtype_in>& k,
                                  const std::vector<dtype_in>& v, size_t qo_len, size_t kv_len,
                                  size_t num_heads, size_t head_dim, bool causal = true,
                                  QKVLayout layout = QKVLayout::kHND,
                                  RotaryMode rotary_mode = RotaryMode::kNone,
                                  float rope_scale = 1.f, float rope_theta = 1e4) {
  assert(qo_len <= kv_len);
  float sm_scale = 1.f / std::sqrt(float(head_dim));
  std::vector<dtype_out> o(qo_len * num_heads * head_dim);
  std::vector<float> att(kv_len);
  std::vector<float> q_rotary_local(head_dim);
  std::vector<float> k_rotary_local(head_dim);
  SWITCH_LAYOUT(layout, LAYOUT, {
    tensor_info_t<LAYOUT> qkv_info(qo_len, kv_len, num_heads, head_dim);
    for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
      for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
        float max_val = -5e4;
        if (rotary_mode == RotaryMode::kLlama) {
          q_rotary_local = std::move(cpu_reference::apply_llama_rope(
              q.data() + qkv_info.get_qo_elem_offset(q_idx, head_idx, 0), head_dim,
              q_idx + kv_len - qo_len, rope_scale, rope_theta));
        }
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] = 0.;
          switch (rotary_mode) {
            case RotaryMode::kNone: {
              for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                att[kv_idx] += float(q[qkv_info.get_qo_elem_offset(q_idx, head_idx, feat_idx)]) *
                               float(k[qkv_info.get_kv_elem_offset(kv_idx, head_idx, feat_idx)]) *
                               sm_scale;
              }
              break;
            }
            case RotaryMode::kLlama: {
              k_rotary_local = std::move(cpu_reference::apply_llama_rope(
                  k.data() + qkv_info.get_kv_elem_offset(kv_idx, head_idx, 0), head_dim, kv_idx,
                  rope_scale, rope_theta));
              for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                att[kv_idx] += q_rotary_local[feat_idx] * k_rotary_local[feat_idx] * sm_scale;
              }
              break;
            }
            default: {
              std::cerr << "Unsupported rotary mode." << std::endl;
              abort();
            }
          }
          // apply mask
          if (causal && q_idx - qo_len < kv_idx - kv_len) {
            att[kv_idx] = -5e4;
          }
          max_val = std::max(max_val, att[kv_idx]);
        }
        // exp minus max
        float denom = 0;
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] = std::exp(att[kv_idx] - max_val);
          denom += att[kv_idx];
        }

        // divide by denom
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] /= denom;
        }

        for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
          float o_float = 0.;
          for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
            o_float +=
                att[kv_idx] * float(v[qkv_info.get_kv_elem_offset(kv_idx, head_idx, feat_idx)]);
          }
          o[qkv_info.get_qo_elem_offset(q_idx, head_idx, feat_idx)] = dtype_out(o_float);
        }
      }
    }
  });
  return std::move(o);
}

/*!
 * \brief A paged key-value cache for CPU reference.
 * \note The input layout: [batch, num_head, head_dim]
 * \note which is the natural output of dense layer.
*/
template <typename T, typename IdxType>
void append_paged_kv_cache(
  paged_kv_t<T, IdxType> page_cpu,
  const std::vector<std::vector<uint8_t>>& keys,
  const std::vector<std::vector<uint8_t>>& values,
  const std::vector<std::vector<half2>>& keys_param,
  const std::vector<std::vector<half2>>& values_param,
  const std::vector<IdxType>& append_indptr
) {
  size_t batch_size = page_cpu.batch_size;
  size_t num_heads = page_cpu.num_heads;
  size_t head_dim = page_cpu.head_dim;
  size_t page_size = page_cpu.page_size;
  for (size_t i = 0; i < batch_size; ++i) {
    const std::vector<uint8_t>& ki = keys[i];
    const std::vector<uint8_t>& vi = values[i];
    const std::vector<half2>& ki_param = keys_param[i];
    const std::vector<half2>& vi_param = values_param[i];

    size_t append_seq_len = append_indptr[i + 1] - append_indptr[i];
    size_t num_pages_i = page_cpu.indptr[i + 1] - page_cpu.indptr[i];
    size_t seq_len = (num_pages_i - 1) * page_size + page_cpu.last_page_offset[i];

    assert(append_seq_len <= seq_len);
    size_t append_start = seq_len - append_seq_len;

    for (size_t j = 0; j < append_seq_len; ++j) {
      size_t page_seq_idx = j + append_start;
      size_t page_idx = page_cpu.indices[page_cpu.indptr[i] + page_seq_idx / page_size];
      size_t entry_idx = page_seq_idx % page_size;
      for (size_t h = 0; h < num_heads; ++h) {
        std::copy(
          ki.begin() + static_cast<size_t>((j * num_heads + h) * head_dim * quant::size_of_type<T>() / sizeof(T)),
          ki.begin() + static_cast<size_t>((j * num_heads + h + 1) * head_dim * quant::size_of_type<T>() / sizeof(T)),
          reinterpret_cast<uint8_t*>(quant::get_ptr(page_cpu.data, page_cpu.get_k_elem_offset(page_idx, h, entry_idx, 0)))
        );
        std::copy(
          vi.begin() + static_cast<size_t>((j * num_heads + h) * head_dim * quant::size_of_type<T>() / sizeof(T)),
          vi.begin() + static_cast<size_t>((j * num_heads + h + 1) * head_dim * quant::size_of_type<T>() / sizeof(T)),
          reinterpret_cast<uint8_t*>(quant::get_ptr(page_cpu.data, page_cpu.get_v_elem_offset(page_idx, h, entry_idx, 0)))
        );
        std::copy(
          ki_param.begin() + (j * num_heads + h),
          ki_param.begin() + (j * num_heads + h + 1),
          quant::get_ptr(page_cpu.param, page_cpu.get_param_k_elem_offset(page_idx, h, entry_idx))
        );
        std::copy(
          vi_param.begin() + (j * num_heads + h),
          vi_param.begin() + (j * num_heads + h + 1),
          quant::get_ptr(page_cpu.param, page_cpu.get_param_v_elem_offset(page_idx, h, entry_idx))
        );
      }
    }
  }
}

template <typename QType, typename KVType, typename OType>
std::vector<OType> single_quantize_mha(
  const std::vector<QType>& q,
  const std::vector<uint8_t>& k,
  const std::vector<uint8_t>& v,
  const std::vector<half2>& k_param,
  const std::vector<half2>& v_param,
  size_t qo_len,
  size_t kv_len,
  size_t num_heads, size_t head_dim, bool causal = true,
  QKVLayout layout = QKVLayout::kHND,
  RotaryMode rotary_mode = RotaryMode::kNone,
  float rope_scale = 1.f, float rope_theta = 1e4
){
  std::vector<float> k_dequant(num_heads * kv_len * head_dim);
  std::vector<float> v_dequant(num_heads * kv_len * head_dim);
  std::vector<float> q_dequant(q.size());
  switch(layout){
    case QKVLayout::kHND:
      for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          float k_scale = __half2float(k_param[head_idx * kv_len + kv_idx].x);
          float k_zero = __half2float(k_param[head_idx * kv_len + kv_idx].y);
          float v_scale = __half2float(v_param[head_idx * kv_len + kv_idx].x);
          float v_zero = __half2float(v_param[head_idx * kv_len + kv_idx].y);
          for (size_t feat_idx = 0; feat_idx < head_dim / 2; ++feat_idx){
            uint8_t packedK = k[head_idx * kv_len * head_dim / 2 + kv_idx * head_dim / 2 + feat_idx];
            k_dequant[head_idx * kv_len * head_dim + kv_idx * head_dim + 2 * feat_idx] = (static_cast<float>(packedK & 0xf) * k_scale - k_zero);
            k_dequant[head_idx * kv_len * head_dim + kv_idx * head_dim + 2 * feat_idx + 1] = (static_cast<float>(packedK >> 4) * k_scale - k_zero);

            uint8_t packedV = v[head_idx * kv_len * head_dim / 2 + kv_idx * head_dim / 2 + feat_idx];
            v_dequant[head_idx * kv_len * head_dim + kv_idx * head_dim + 2 * feat_idx] = (static_cast<float>(packedV & 0xf) * v_scale - v_zero);
            v_dequant[head_idx * kv_len * head_dim + kv_idx * head_dim + 2 * feat_idx + 1] = (static_cast<float>(packedV >> 4) * v_scale - v_zero);
          }
        }
      }
      break;
    case QKVLayout::kNHD:
      for(size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx){
        for(size_t head_idx = 0; head_idx < num_heads; ++head_idx){
          float k_scale = __half2float(k_param[kv_idx * num_heads + head_idx].x);
          float k_zero = __half2float(k_param[kv_idx * num_heads + head_idx].y);
          float v_scale = __half2float(v_param[kv_idx * num_heads + head_idx].x);
          float v_zero = __half2float(v_param[kv_idx * num_heads + head_idx].y);
          for(size_t feat_idx = 0; feat_idx < head_dim / 2; ++feat_idx){
            uint8_t packedK = k[kv_idx * num_heads * head_dim / 2 + head_idx * head_dim / 2 + feat_idx];
            k_dequant[kv_idx * num_heads * head_dim + head_idx * head_dim + 2 * feat_idx] = (static_cast<float>(packedK & 0xf) * k_scale - k_zero);
            k_dequant[kv_idx * num_heads * head_dim + head_idx * head_dim + 2 * feat_idx + 1] = (static_cast<float>(packedK >> 4) * k_scale - k_zero);

            uint8_t packedV = v[kv_idx * num_heads * head_dim / 2 + head_idx * head_dim / 2 + feat_idx];
            v_dequant[kv_idx * num_heads * head_dim + head_idx * head_dim + 2 * feat_idx] = (static_cast<float>(packedV & 0xf) * v_scale - v_zero);
            v_dequant[kv_idx * num_heads * head_dim + head_idx * head_dim + 2 * feat_idx + 1] = (static_cast<float>(packedV >> 4) * v_scale - v_zero);
          }
        }
      }
      break;
  }

  for (size_t i = 0; i < q.size(); ++i) {
    q_dequant[i] = __half2float(q[i]);
  }

  return single_mha<float, OType>(q_dequant, k_dequant, v_dequant, qo_len, kv_len, num_heads, head_dim, causal, layout, rotary_mode, rope_scale, rope_theta);
}

}  // namespace cpu_reference