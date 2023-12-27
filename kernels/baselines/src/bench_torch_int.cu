#include <linear.h>
#include <nvbench/nvbench.cuh>

void bench_torch_int(nvbench::state &state) {
  size_t bsz = state.get_int64("bsz");

  constexpr int hidden_dim = 4096;
  
  // Init torch::Tensor with int8_t
  auto hidden_states = torch::randint(63, {(int)bsz, hidden_dim}, torch::kInt8).to(torch::kCUDA);
  auto weight = torch::randint(63, {hidden_dim, hidden_dim}, torch::kInt8).to(torch::kCUDA);
  auto bias = torch::randint(63, {hidden_dim}, torch::kInt8).to(torch::kCUDA);
  float alpha = 0.5f;
  float beta = 0.5f;

  state.add_global_memory_reads<int8_t>(
    hidden_states.numel() + weight.numel() + bias.numel(),
    "Read"
  );
  state.add_global_memory_writes<int8_t>(
    hidden_states.numel(),
    "Write"
  );
  // Provide Computation throughput information
  state.add_element_count(
    bsz * hidden_dim * hidden_dim * 2
  );

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    linear_a8_w8_b8_o8(hidden_states, weight, bias, alpha, beta);
  });
}

NVBENCH_BENCH(                                                                   
    bench_torch_int
).set_name("bench_torch_int")            
.add_int64_axis("bsz", {1,4,8,16,32,128,256,512,1024,2048,4096});