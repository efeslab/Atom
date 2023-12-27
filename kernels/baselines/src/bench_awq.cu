#include <gemv_cuda.h>
#include <nvbench/nvbench.cuh>

void bench_awq(nvbench::state &state) {
  size_t bsz = state.get_int64("bsz");

  constexpr int hidden_dim = 4096;
  constexpr int group_size = 128;
  // Init torch::Tensor with corresponding size
  // Ref: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/gemv_cuda.cu#L189

  auto hidden_states = torch::randn({(int)bsz, hidden_dim}, torch::kFloat16).to(torch::kCUDA);
  auto weight = torch::randint(128, {hidden_dim, hidden_dim / 8}, torch::kInt32).to(torch::kCUDA);
  auto scales = torch::randn({hidden_dim, hidden_dim / group_size}, torch::kFloat16).to(torch::kCUDA);
  auto zeros = torch::randint(128, {hidden_dim, hidden_dim / group_size / 8}, torch::kInt32).to(torch::kCUDA);

  state.add_global_memory_reads<int16_t>(
    hidden_states.numel() + scales.numel(),
    "Read"
  );
  state.add_global_memory_reads<int32_t>(
    weight.numel() + zeros.numel(),
    "Read"
  );
  state.add_global_memory_writes<int16_t>(
    hidden_states.numel(),
    "Write"
  );
  // Provide Computation throughput information
  state.add_element_count(
    bsz * hidden_dim * hidden_dim * 2
  );

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    gemv_forward_cuda(
      hidden_states,
      weight,
      scales,
      zeros,
      group_size
    );
  });
}

NVBENCH_BENCH(                                                                   
    bench_awq
).set_name("bench_awq")            
.add_int64_axis("bsz", {1,4,8,16,32,128,256,512,1024,2048,4096});