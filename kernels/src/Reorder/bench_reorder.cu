#include <thrust/device_vector.h>
#include <nvbench/nvbench.cuh>
#include <Reorder/Reorder.cuh>
#include <vector>
#include <algorithm>


void bench_reorder(nvbench::state &state) {
  size_t bsz = state.get_int64("bsz");

  constexpr int hidden_dim = 4096;
  constexpr int group_size = 128; // Here Keeper size fix as group_size.

  // Allocate input data:
  thrust::device_vector<half> hidden_states(bsz * hidden_dim);
  thrust::device_vector<int8_t> o_outliers(bsz * group_size);
  thrust::device_vector<int8_t> o_norms(bsz * (hidden_dim - group_size) / 2);
  thrust::device_vector<half> outlier_scales(SCALE_SIZE_A(bsz));
  thrust::device_vector<half> norm_scales(SCALE_SIZE_A(bsz) * (hidden_dim / group_size - 1));

  std::vector<int16_t> reorder_index_h(hidden_dim);
  for(int i = 0; i < hidden_dim; i++) {
    reorder_index_h[i] = i;
  }
  std::random_shuffle(reorder_index_h.begin(), reorder_index_h.end());
  thrust::device_vector<int16_t> reorder_index(reorder_index_h);

  // Provide throughput information:
  state.add_global_memory_reads<half>(
    hidden_states.size() + reorder_index.size(),
    "Read"
  );
  state.add_global_memory_writes<half>(
    outlier_scales.size() + norm_scales.size(),
    "Write"
  );
  state.add_global_memory_writes<int8_t>(
    o_outliers.size() + o_norms.size(),
    "Write"
  );

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
    timer.start();
    run_reorder_fp16_i4<group_size, hidden_dim>(
      thrust::raw_pointer_cast(hidden_states.data()),
      bsz,
      thrust::raw_pointer_cast(reorder_index.data()),
      thrust::raw_pointer_cast(o_outliers.data()),
      thrust::raw_pointer_cast(o_norms.data()),
      thrust::raw_pointer_cast(outlier_scales.data()),
      thrust::raw_pointer_cast(norm_scales.data())
    );
    timer.stop();
  });
}

NVBENCH_BENCH(                                                                   
    bench_reorder
).set_name("bench_reorder")            
.add_int64_axis("bsz", {1,4,8,16,32,128,256,512,1024});