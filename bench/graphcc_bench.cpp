#include <benchmark/benchmark.h>
#include <xxhash.h>
#include <iostream>
#include <unistd.h>
#include <fstream>

#include "binary_graph_stream.h"
#include "bucket.h"
#include "test/sketch_constructors.h"

constexpr uint64_t KB   = 1024;
constexpr uint64_t MB   = KB * KB;
constexpr uint64_t seed = 374639;

// Linux-only, flush the filesystem cache
// requires sudo privileges :(
static void flush_filesystem_cache() {
  sync();
  std::ofstream drop("/proc/sys/vm/drop_caches");
  drop << "3" << std::endl;
}

// Test the speed of reading all the data in the kron16 graph stream
static void BM_FileIngest(benchmark::State &state) {
  double total_updates = 0;
  for (auto _ : state) {
    BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_16_stream_binary", state.range(0));

    uint64_t m = stream.edges();
    GraphUpdate upd;
    while (m--) {
      benchmark::DoNotOptimize(upd = stream.get_edge());
    }
    total_updates += stream.edges();
  }
  flush_filesystem_cache();
  state.counters["Ingestion_Rate"] = benchmark::Counter(total_updates, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_FileIngest)->RangeMultiplier(4)->Range(KB, MB);

// Test the speed of hashing using a variety of hash methods
// The arguments to these benchmarks is the number of hashes to perform serially
static void BM_Hash_XXHash64(benchmark::State &state) {
  double total_calls = 0;
  for (auto _ : state) {
    uint64_t num_hashes = state.range(0);
    for (uint64_t i = 0; i < num_hashes; i++) {
      benchmark::DoNotOptimize(XXH3_64bits_withSeed(&i, sizeof(uint64_t), seed));
    }
    total_calls += num_hashes;
  }
  state.counters["Hashes"] = benchmark::Counter(total_calls, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXHash64)->Arg(1)->Arg(100)->Arg(10000); 

static void BM_Hash_bucket(benchmark::State &state) {
  double total_calls = 0;
  for (auto _ : state) {
    uint64_t num_hashes = state.range(0);
    for (uint64_t i = 0; i < num_hashes; i++) {
      benchmark::DoNotOptimize(Bucket_Boruvka::col_index_hash(i, seed));
    }
    total_calls += num_hashes;
  }
  state.counters["Hashes"] = benchmark::Counter(total_calls, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_bucket)->Arg(1)->Arg(100)->Arg(10000);


// Benchmark the speed of updating sketches both serially and in batch mode
static void BM_Sketch_Update(benchmark::State &state) {
  constexpr size_t upd_per_sketch = 10000;
  constexpr size_t num_sketches   = 1000;
  double total_updates = 0;
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
      updates[i].push_back(rand() % vec_size);
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
    total_updates += num_sketches * upd_per_sketch;
  }
  state.counters["Update_Rate"] = benchmark::Counter(total_updates, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Update)->RangeMultiplier(4)->Ranges({{KB << 4, MB << 4}, {false, true}});

// Benchmark the speed of querying sketches
static void BM_Sketch_Query(benchmark::State &state) {
  constexpr size_t vec_size     = KB << 5;
  constexpr size_t num_sketches = 100;
  double total_queries = 0;
  double density = ((double)state.range(0)) / 100;

  // initialize sketches
  Sketch::configure(vec_size, 100);
  SketchUniquePtr sketches[num_sketches];
  for (size_t i = 0; i < num_sketches; i++) {
    sketches[i] = makeSketch(seed + i);
  }

  // perform updates
  for (size_t i = 0; i < num_sketches; i++) {
    for (size_t j = 0; j < vec_size * density; j++) {
      sketches[i]->update(rand() % vec_size);
    }
  }
  std::pair<vec_t, SampleSketchRet> q_ret;

  for (auto _ : state) {
    // perform queries
    for (size_t j = 0; j < num_sketches; j++) {
      benchmark::DoNotOptimize(q_ret = sketches[j]->query());
      sketches[j]->reset_queried();
    }
    total_queries += num_sketches;
  }
  state.counters["Query_Rate"] = benchmark::Counter(total_queries, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Query)->DenseRange(10, 70, 10);

BENCHMARK_MAIN();
