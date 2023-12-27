cmake_minimum_required(VERSION 3.18)

project(baseline CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set for Torch-int
set(CUDA_ARCH 890)
add_definitions(-DCUDA_ARCH=${CUDA_ARCH})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(Thrust REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(${PROJECT_SOURCE_DIR}/../3rdparty/nvbench ${CMAKE_BINARY_DIR}/nvbench)

message(STATUS "Compile torch-int Linear kernel benchmark.")
set(TORCH_INT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/torch-int/torch_int/kernels/include)
add_executable(bench_torch_int ${PROJECT_SOURCE_DIR}/src/bench_torch_int.cu ${TORCH_INT_INCLUDE_DIR}/../linear.cu)
target_include_directories(bench_torch_int PRIVATE ${TORCH_INT_INCLUDE_DIR})
target_include_directories(bench_torch_int PRIVATE ${PROJECT_SOURCE_DIR}/../3rdparty/torch-int/submodules/cutlass/tools/util/include)
target_include_directories(bench_torch_int PRIVATE ${PROJECT_SOURCE_DIR}/../3rdparty/torch-int/submodules/cutlass/include)
target_include_directories(bench_torch_int PRIVATE ${PROJECT_SOURCE_DIR}/../3rdparty/nvbench)
target_link_libraries(bench_torch_int PRIVATE nvbench::main "${TORCH_LIBRARIES}")

message(STATUS "Compile AWQ 4-bit Linear kernel benchmark.")
set(AWQ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/llm-awq/awq/kernels/csrc/quantization)
add_executable(bench_awq ${PROJECT_SOURCE_DIR}/src/bench_awq.cu ${AWQ_INCLUDE_DIR}/gemv_cuda.cu)
target_include_directories(bench_awq PRIVATE ${AWQ_INCLUDE_DIR})
target_include_directories(bench_awq PRIVATE /usr/include/python3.10/)
target_include_directories(bench_awq PRIVATE ${PROJECT_SOURCE_DIR}/../3rdparty/nvbench)
target_compile_options(bench_awq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>) 
target_link_libraries(bench_awq PRIVATE nvbench::main "${TORCH_LIBRARIES}")