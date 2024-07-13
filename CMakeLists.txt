cmake_minimum_required(VERSION 3.15)
project(GraphZeppelin LANGUAGES CXX CUDA)

include (FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(CMAKE_CUDA_ARCHITECTURES "70")

# Make the default build type Release. If user or another
# project sets a different value than use that
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to default -- Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()
message(STATUS "GraphZeppelin Build Type: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
elseif(STATUS "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  message(STATUS "${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)
#add_compile_options(-fsanitize=undefined)
#add_link_options(-fsanitize=undefined)

# Check if this project is the top directory or build type is Debug
# If so, build executables, otherwise, only build libraries
get_directory_property(not_root PARENT_DIRECTORY)
if (not_root AND "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(BUILD_EXE OFF)
else()
  set(BUILD_EXE ON)
  message (STATUS "GraphZeppelin building executables")
endif()

# Print out information of CUDA compiler
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")


# Get GutterTree Project
FetchContent_Declare(
  GutterTree

  GIT_REPOSITORY  https://github.com/GraphStreamingProject/GutterTree.git
  GIT_TAG         main
)

# Get StreamingUtilities
FetchContent_Declare(
  StreamingUtilities

  GIT_REPOSITORY  https://github.com/GraphStreamingProject/StreamingUtilities.git
  GIT_TAG         main
)

# Get VieCut
FetchContent_Declare(
  VieCut

  GIT_REPOSITORY  https://github.com/etwest/VieCut.git
  GIT_TAG         master
)

# Get tlx
FetchContent_Declare(
  tlx

  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         master
)

if (BUILD_BENCH)
  # Get Google Benchmark
  FetchContent_Declare(
    benchmark

    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG        v1.6.1
  )
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

  FetchContent_MakeAvailable(benchmark)
endif()

FetchContent_MakeAvailable(GutterTree StreamingUtilities VieCut tlx)

# AVAILABLE COMPILATION DEFINITIONS:
# VERIFY_SAMPLES_F   Use a deterministic connected-components
#                    algorithm to verify post-processing.
# NO_EAGER_DSU       Do not use the eager DSU query optimization
#                    if this flag is present.
# L0_SAMPLING        Run the CubeSketch l0 sampling algorithm
#                    to ensure that we sample uniformly.
#                    Otherwise, run a support finding algorithm.
#
# Example:
# cmake -DCMAKE_CXX_FLAGS="-DL0_SAMPLING" ..

add_library(CUDALibrary
  src/cuda_library.cu
  src/cuda_kernel.cu
  src/cc_gpu_sketch_alg.cpp
  src/mc_gpu_sketch_alg.cpp)
  #src/sk_gpu_sketch_alg.cpp)
add_dependencies(CUDALibrary GutterTree StreamingUtilities VieCut tlx)
target_link_libraries(CUDALibrary PUBLIC xxhash GutterTree StreamingUtilities VieCut tlx)
target_include_directories(CUDALibrary PUBLIC include/)
target_compile_options(CUDALibrary PUBLIC -lineinfo) # -lineinfo: For profiling
#target_compile_options(CUDALibrary PUBLIC -expt-relaxed-constexpr)

add_library(GraphZeppelin
  src/cc_sketch_alg.cpp
  src/mc_sketch_alg.cpp
  src/mc_subgraph.cpp
  src/return_types.cpp
  src/driver_configuration.cpp
  src/cc_alg_configuration.cpp
  src/sketch.cpp
  src/util.cpp)
add_dependencies(GraphZeppelin GutterTree StreamingUtilities VieCut tlx)
target_link_libraries(GraphZeppelin PUBLIC xxhash GutterTree StreamingUtilities VieCut tlx)
target_include_directories(GraphZeppelin PUBLIC include/)
target_compile_options(GraphZeppelin PUBLIC -fopenmp)
target_link_options(GraphZeppelin PUBLIC -fopenmp)
target_compile_definitions(GraphZeppelin PUBLIC XXH_INLINE_ALL)

add_library(GraphZeppelinVerifyCC
  src/cc_sketch_alg.cpp
  src/return_types.cpp
  src/driver_configuration.cpp
  src/cc_alg_configuration.cpp
  src/sketch.cpp
  src/util.cpp
  test/util/graph_verifier.cpp)
add_dependencies(GraphZeppelinVerifyCC GutterTree StreamingUtilities VieCut)
target_link_libraries(GraphZeppelinVerifyCC PUBLIC xxhash GutterTree StreamingUtilities VieCut)
target_include_directories(GraphZeppelinVerifyCC PUBLIC include/ include/test/)
target_compile_options(GraphZeppelinVerifyCC PUBLIC -fopenmp)
target_link_options(GraphZeppelinVerifyCC PUBLIC -fopenmp)
target_compile_definitions(GraphZeppelinVerifyCC PUBLIC XXH_INLINE_ALL VERIFY_SAMPLES_F)

if (BUILD_EXE)
  add_executable(tests
    test/test_runner.cpp
    test/cc_alg_test.cpp
    test/sketch_test.cpp
    test/dsu_test.cpp
    test/util_test.cpp
    test/util/graph_verifier_test.cpp)
  add_dependencies(tests GraphZeppelinVerifyCC)
  target_link_libraries(tests PRIVATE GraphZeppelinVerifyCC)

  add_executable(statistical_sketch_test
    tools/sketch_testing.cpp)
  add_dependencies(statistical_sketch_test GraphZeppelinVerifyCC)
  target_link_libraries(statistical_sketch_test PRIVATE GraphZeppelinVerifyCC)
 

  # executable for processing a binary graph stream
  add_executable(process_stream
    tools/process_stream.cpp)
  target_link_libraries(process_stream PRIVATE GraphZeppelin)

  # executable for performing in depth correctness testing
  add_executable(test_correctness
    tools/test_correctness.cpp)
  target_link_libraries(test_correctness PRIVATE GraphZeppelinVerifyCC)

  # executable for processing a binary graph stream for CUDA
  add_executable(cuda_process_stream
    tools/cuda_process_stream.cu)
  target_link_libraries(cuda_process_stream PRIVATE GraphZeppelin CUDALibrary)

  #add_executable(single_kernel_stream
  #  tools/single_kernel_stream.cu)
  #target_link_libraries(single_kernel_stream PRIVATE GraphZeppelin CUDALibrary)

  add_executable(xxhash_test
    test/xxhash_test.cu)
  target_link_libraries(xxhash_test PRIVATE GraphZeppelin CUDALibrary)

  add_executable(min_cut
    tools/min_cut.cu)
  target_link_libraries(min_cut PRIVATE GraphZeppelin CUDALibrary)

  add_executable(adjlist_test
    tools/adjlist_test.cpp)
  target_link_libraries(adjlist_test PRIVATE GraphZeppelin CUDALibrary)

  add_executable(generate_stream
    tools/stream_generators/non_trivial_stream_generator.cpp
  )
  target_link_libraries(generate_stream PRIVATE GraphZeppelin)

  add_executable(metis_converter
    tools/converter/metis_converter.cpp
  )
  target_link_libraries(metis_converter PRIVATE GraphZeppelin)

  add_executable(stream_converter
    tools/converter/stream_converter.cpp
  )
  target_link_libraries(stream_converter PRIVATE GraphZeppelin)

  add_executable(stream_shuffler
    tools/others/stream_shuffler.cpp
  )
  target_link_libraries(stream_shuffler PRIVATE GraphZeppelin)
endif()

if (BUILD_BENCH)
  add_executable(bench_cc
    tools/benchmark/graphcc_bench.cpp)
  add_dependencies(bench_cc GraphZeppelin benchmark)
  target_link_libraries(bench_cc GraphZeppelin benchmark::benchmark xxhash)
endif()

