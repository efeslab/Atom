#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace utils {

template <typename T>
void vec_normal_(std::vector<T>& vec, float mean = 0.f, float std = 1.f) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <>
void vec_normal_<half2>(std::vector<half2>& vec, float mean, float std) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i].x = __float2half(d(gen));
    vec[i].y = __float2half(d(gen));
  }
}

template <>
void vec_normal_<uint8_t>(std::vector<uint8_t>& vec, float mean, float std) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = (uint8_t)((uint32_t)(d(gen) * 1000) % 256);
  }
}

template <>
void vec_normal_(std::vector<flashinfer::quant::__precision__s4>& vec, float mean, float std){
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    reinterpret_cast<int8_t*>(vec.data())[i] = (int8_t)((int32_t)(d(gen) * 1000) % 256);
  }
}

template <typename T>
void vec_zero_(std::vector<T>& vec) {
  std::fill(vec.begin(), vec.end(), T(0));
}

template <>
void vec_zero_(std::vector<flashinfer::quant::__precision__s4>& vec) {
  // vec.size() is correct since we allocate memory by bytes.
  memset((void *)vec.data(), 0, vec.size());
}

template <>
void vec_zero_<half2>(std::vector<half2>& vec) {
  memset((void *)vec.data(), 0, vec.size() * sizeof(half2));
}

template <typename T>
void vec_fill_(std::vector<T>& vec, T val) {
  std::fill(vec.begin(), vec.end(), val);
}

template <typename T>
size_t vec_bytes(const T& vec) {
  return vec.size() * sizeof(typename T::value_type);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

}  // namespace utils
