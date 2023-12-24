#include <thrust/device_vector.h>
#include <nvbench/nvbench.cuh>
#include <GEMM/Dense_layer_gemm_i4_o16.cuh>

void bench_DenseLayerGEMM_i4(nvbench::state &state) {
  size_t bsz = state.get_int64("bsz");
  size_t keeper_size = state.get_int64("keeper_size");
  size_t hidden_dim = state.get_int64("hidden_dim");

  size_t TEST_M_GLOBAL = bsz;
  size_t TEST_K_GLOBAL = hidden_dim - keeper_size;
  size_t TEST_N_GLOBAL = hidden_dim;

  // Allocate input data:
  thrust::device_vector<uint8_t> A(TEST_M_GLOBAL * TEST_K_GLOBAL / 2);
  thrust::device_vector<uint8_t> B(TEST_N_GLOBAL * TEST_K_GLOBAL / 2);
  thrust::device_vector<half2> A_s(SCALE_SIZE_A(TEST_M_GLOBAL) * ((TEST_K_GLOBAL + GROUP_SIZE - 1) / GROUP_SIZE));
  thrust::device_vector<half2> B_s(SCALE_PACKING_B(TEST_N_GLOBAL) * ((TEST_K_GLOBAL + GROUP_SIZE - 1) / GROUP_SIZE));

  thrust::device_vector<uint8_t> A_keeper(TEST_M_GLOBAL * keeper_size);
  thrust::device_vector<uint8_t> B_keeper(TEST_N_GLOBAL * keeper_size);
  thrust::device_vector<half2> A_keeper_s(SCALE_SIZE_A(TEST_M_GLOBAL));
  thrust::device_vector<half2> B_keeper_s(SCALE_PACKING_B(TEST_N_GLOBAL));

  thrust::device_vector<half> D(TEST_M_GLOBAL * TEST_N_GLOBAL);
  // Provide throughput information:
  state.add_global_memory_reads<uint8_t>(
    A.size() + B.size() + A_keeper.size() + B_keeper.size(),
    "Read"
  );
  state.add_global_memory_reads<half2>(
    A_s.size() + B_s.size() + A_keeper_s.size() + B_keeper_s.size(),
    "Read"
  );
  state.add_global_memory_writes<half>(
    D.size(),
    "Write"
  );
  // Provide Computation throughput information
  state.add_element_count(
    TEST_M_GLOBAL * TEST_N_GLOBAL * (TEST_K_GLOBAL + keeper_size) * 2
  );

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
    timer.start();
    DenseLayerGEMM_i4_o16(
      thrust::raw_pointer_cast(A.data()),
      thrust::raw_pointer_cast(B.data()),
      (uint8_t*)thrust::raw_pointer_cast(A_s.data()),
      (uint8_t*)thrust::raw_pointer_cast(B_s.data()),
      thrust::raw_pointer_cast(A_keeper.data()),
      thrust::raw_pointer_cast(B_keeper.data()),
      (uint8_t*)thrust::raw_pointer_cast(A_keeper_s.data()),
      (uint8_t*)thrust::raw_pointer_cast(B_keeper_s.data()),
      thrust::raw_pointer_cast(D.data()),
      TEST_M_GLOBAL,
      TEST_N_GLOBAL,
      TEST_K_GLOBAL
    );
    timer.stop();
  });
}

NVBENCH_BENCH(                                                                   
    bench_DenseLayerGEMM_i4
).set_name("Dense_layer_gemm_i4_o16")                       
.add_int64_axis("bsz", {16,32,64,128,256,512,1024,2048,4096})
.add_int64_axis("keeper_size", {128})
.add_int64_axis("hidden_dim", {4096});