#include "flashinfer_config.h"
#include "flashinfer_impl.cuh"

template void FlashInferBatchDecodeKernel_i4<128>(
    nv_half* o, nv_half* q, void* kv_data, nv_half2* kv_param,
    int32_t* kv_indptr, int32_t* kv_indicies, int32_t* last_page_offset,
    int num_layers, int layer_idx, int num_heads, int page_size,
    int batch_size);

template void FlashInferInitKvKernel_i4<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int32_t* seqlen_indptr, int num_layers,
    int layer_idx, int num_heads, int page_size, int batch_size);

template void FlashInferAppendKvKernel_i4<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int num_layers, int layer_idx, int num_heads,
    int page_size, int batch_size);
