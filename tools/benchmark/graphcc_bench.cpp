#include <benchmark/benchmark.h>
#include <xxhash.h>
#include <iostream>
#include <unistd.h>
#include <fstream>

#include "binary_graph_stream.h"
#include "bucket.h"
#include "l0_sampling/pmshash.h"
#include "test/sketch_constructors.h"

constexpr uint64_t KB   = 1024;
constexpr uint64_t MB   = KB * KB;
constexpr uint64_t seed = 374639;

// Linux-only, flush the filesystem cache
// requires sudo privileges :(
static void flush_filesystem_cache() {
  sync();
  std::ofstream drop("/proc/sys/vm/drop_caches");
  if (drop.is_open()) {
    drop << "3" << std::endl;
  }
  else {
    std::cout << "WARNING: could not drop filesystem cache. BM_FileIngest will be inaccurate. ";
    std::cout << "Running as root may be required." << std::endl;
  }
}

// Test the speed of reading all the data in the kron16 graph stream
static void BM_FileIngest(benchmark::State &state) {
  // determine the number of edges in the graph
  uint64_t num_edges;
  {
    BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_16_stream_binary", 1024);
    num_edges = stream.edges();
  }

  // flush fs cache
  flush_filesystem_cache();
  
  // perform benchmark
  for (auto _ : state) {
    BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_16_stream_binary", state.range(0));

    uint64_t m = stream.edges();
    GraphUpdate upd;
    while (m--) {
      benchmark::DoNotOptimize(upd = stream.get_edge());
    }
  }
  state.counters["Ingestion_Rate"] = benchmark::Counter(state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_FileIngest)->RangeMultiplier(2)->Range(KB << 2, MB / 4);

// Test the speed of hashing using a method that loops over seeds and a method that 
// batches by seed
// The argument to this benchmark is the number of hashes to batch
static void BM_Hash_XXH64(benchmark::State &state) { 
  uint64_t num_seeds = 8;
  uint64_t num_hashes = state.range(0);
  uint64_t output;
  for (auto _ : state) {
    for (uint64_t h = 0; h < num_seeds; h++) {
      for (uint64_t i = 0; i < num_hashes; i++) {
        benchmark::DoNotOptimize(output = XXH64(&i, sizeof(uint64_t), seed + h));
      }
    }
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations() * num_hashes, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH64)->Arg(1)->Arg(100)->Arg(10000);

static void BM_Hash_XXH3_64(benchmark::State &state) { 
  uint64_t num_seeds = 8;
  uint64_t num_hashes = state.range(0);
  uint64_t output;
  for (auto _ : state) {
    for (uint64_t h = 0; h < num_seeds; h++) {
      for (uint64_t i = 0; i < num_hashes; i++) {
        benchmark::DoNotOptimize(output = XXH3_64bits_withSeed(&i, sizeof(uint64_t), seed + h));
      }
    }
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations() * num_hashes, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH3_64)->Arg(1)->Arg(100)->Arg(10000);

static void BM_Hash_XXPMS64(benchmark::State &state) {
  uint64_t num_seeds = 8;
  uint64_t num_hashes = state.range(0);
  uint64_t output;
  for (auto _ : state) {
    for (uint64_t h = 0; h < num_seeds; h++) {
      for (uint64_t i = 0; i < num_hashes; i++) {
        benchmark::DoNotOptimize(output = XXPMS64(i, seed + h));
      }
    }
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations() * num_hashes, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXPMS64)->Arg(1)->Arg(100)->Arg(10000);

static void BM_Hash_bucket(benchmark::State &state) {
  uint64_t num_seeds = 8;
  uint64_t num_hashes = state.range(0);
  uint64_t output;
  for (auto _ : state) {
    for (uint64_t h = 0; h < num_seeds; h++) {
      for (uint64_t i = 0; i < num_hashes; i++) {
        benchmark::DoNotOptimize(output = Bucket_Boruvka::col_index_hash(i, seed + h));
      }
    }
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations() * num_hashes, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_bucket)->Arg(1)->Arg(100)->Arg(10000);

// Benchmark the speed of updating sketches both serially and in batch mode
static void BM_Sketch_Update(benchmark::State &state) {
  constexpr size_t upd_per_sketch = 10000;
  constexpr size_t num_sketches   = 1000;
  size_t vec_size = state.range(0);
  // initialize sketches
  Sketch::configure(vec_size, 100);
  SketchUniquePtr sketches[num_sketches];
  for (size_t i = 0; i < num_sketches; i++) {
    sketches[i] = makeSketch(seed + i);
  }

  // initialize updates
  srand(seed);
  std::vector<std::vector<vec_t>> updates;
  for (size_t i = 0; i < num_sketches; i++) {
    updates.emplace_back();
    updates[i].reserve(upd_per_sketch);
    for (size_t j = 0; j < upd_per_sketch; j++) {
      updates[i].push_back(j % vec_size);
    }
  }

  // Test the speed of updating the sketches
  for (auto _ : state) {
    // perform updates
    if (!state.range(1)) {
      // update serially
      for (size_t j = 0; j < upd_per_sketch; j++) {
        for (size_t i = 0; i < num_sketches; i++) {
          sketches[i]->update(updates[i][j]);
        }
      }
    } else {
      // update in batch
      for (size_t i = 0; i < num_sketches; i++) {
        sketches[i]->batch_update(updates[i]);
      }
    }
  }
  state.counters["Update_Rate"] = benchmark::Counter(state.iterations() * upd_per_sketch * num_sketches,
                                                     benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Update)->RangeMultiplier(4)->Ranges({{KB << 4, MB << 4}, {false, true}});

// Benchmark the speed of querying sketches
static void BM_Sketch_Query(benchmark::State &state) {
  constexpr size_t vec_size     = KB << 5;
  constexpr size_t num_sketches = 100;
  double density = ((double)state.range(0)) / 100;

  // initialize sketches
  Sketch::configure(vec_size, 100);
  SketchUniquePtr sketches[num_sketches];
  for (size_t i = 0; i < num_sketches; i++) {
    sketches[i] = makeSketch(seed + i);
  }

  // perform updates (do at least 1)
  for (size_t i = 0; i < num_sketches; i++) {
    for (size_t j = 0; j < vec_size * density + 1; j++) {
      sketches[i]->update(j + 1);
    }
  }
  std::pair<vec_t, SampleSketchRet> q_ret;

  for (auto _ : state) {
    // perform queries
    for (size_t j = 0; j < num_sketches; j++) {
      benchmark::DoNotOptimize(q_ret = sketches[j]->query());
      sketches[j]->reset_queried();
    }
  }
  state.counters["Query_Rate"] = benchmark::Counter(state.iterations() * num_sketches, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Query)->DenseRange(0, 90, 10);

BENCHMARK_MAIN();
