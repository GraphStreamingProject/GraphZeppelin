#include <benchmark/benchmark.h>
#include <unistd.h>
#include <xxhash.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "binary_graph_stream.h"
#include "bucket.h"
#include "dsu.h"
#include "test/sketch_constructors.h"

constexpr uint64_t KB = 1024;
constexpr uint64_t MB = KB * KB;
constexpr uint64_t seed = 374639;

// If this flag is uncommented then run the FileIngestion benchmarks
// #define FILE_INGEST_F

#ifdef FILE_INGEST_F
// Linux-only, flush the filesystem cache
// requires sudo privileges :(
static void flush_filesystem_cache() {
  sync();
  std::ofstream drop("/proc/sys/vm/drop_caches");
  if (drop.is_open()) {
    drop << "3" << std::endl;
  } else {
    std::cout << "WARNING: could not drop filesystem cache. BM_FileIngest will be inaccurate. ";
    std::cout << "Running as root may be required." << std::endl;
  }
}

// Test the speed of reading all the data in the kron16 graph stream
static void BM_FileIngest(benchmark::State& state) {
  // determine the number of edges in the graph
  uint64_t num_edges;
  {
    BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", 1024);
    num_edges = stream.edges();
  }

  // flush fs cache
  flush_filesystem_cache();

  // perform benchmark
  for (auto _ : state) {
    BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", state.range(0));

    uint64_t m = stream.edges();
    GraphUpdate upd;
    while (m--) {
      benchmark::DoNotOptimize(upd = stream.get_edge());
    }
  }
  state.counters["Ingestion_Rate"] = benchmark::Counter(
    state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_FileIngest)->RangeMultiplier(2)->Range(KB << 2, MB / 4)->UseRealTime();

// Test the speed of reading all the data in the kron16 graph stream
static void BM_MTFileIngest(benchmark::State& state) {
  // determine the number of edges in the graph
  uint64_t num_edges;
  {
    BinaryGraphStream_MT stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", 1024);
    num_edges = stream.edges();
  }

  // flush fs cache
  flush_filesystem_cache();

  // perform benchmark
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(state.range(0));

    BinaryGraphStream_MT stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", 32 * 1024);

    auto task = [&]() {
      MT_StreamReader reader(stream);
      GraphUpdate upd;
      do {
        upd = reader.get_edge();
      } while (upd.type != BREAKPOINT);
    };

    for (int i = 0; i < state.range(0); i++) threads.emplace_back(task);
    for (int i = 0; i < state.range(0); i++) threads[i].join();
  }
  state.counters["Ingestion_Rate"] = benchmark::Counter(
    state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_MTFileIngest)->RangeMultiplier(4)->Range(1, 20)->UseRealTime();
#endif // FILE_INGEST_F

static void BM_builtin_ffsl(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ffsl(i++));
    benchmark::DoNotOptimize(__builtin_ffsl(j++));
  }
}
BENCHMARK(BM_builtin_ffsl);

static void BM_builtin_ctzl(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ctzl(i++));
    benchmark::DoNotOptimize(__builtin_ctzl(j++));
  }
}
BENCHMARK(BM_builtin_ctzl);

static void BM_builtin_clzl(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_clzl(i++));
    benchmark::DoNotOptimize(__builtin_clzl(j++));
  }
}
BENCHMARK(BM_builtin_clzl);

// Test the speed of hashing using a method that loops over seeds and a method that
// batches by seed
// The argument to this benchmark is the number of hashes to batch
static void BM_Hash_XXH64(benchmark::State& state) {
  uint64_t input = 100'000;
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(XXH64(&input, sizeof(uint64_t), seed));
  }
  state.counters["Hash Rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH64);

static void BM_Hash_XXH3_64(benchmark::State& state) {
  uint64_t input = 100'000;
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(XXH3_64bits_withSeed(&input, sizeof(uint64_t), seed));
  }
  state.counters["Hash Rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH3_64);

static void BM_index_depth_hash(benchmark::State& state) {
  uint64_t input = 100'000;
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(Bucket_Boruvka::get_index_depth(input, seed, 20));
  }
  state.counters["Hash Rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_index_depth_hash);

static void BM_index_hash(benchmark::State& state) {
  uint64_t input = 100'000;
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(Bucket_Boruvka::get_index_hash(input, seed));
  }
  state.counters["Hash Rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_index_hash);

static void BM_update_bucket(benchmark::State& state) {
  vec_t a = 0;
  vec_hash_t c = 0;
  vec_t input = 0x0EADBEEF;
  vec_hash_t checksum = 0x0EEDBEEF;

  for (auto _ : state) {
    ++input;
    ++checksum;
    Bucket_Boruvka::update(a, c, input, checksum);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_update_bucket);

// Benchmark the speed of updating sketches both serially and in batch mode
static void BM_Sketch_Update(benchmark::State& state) {
  size_t vec_size = state.range(0);
  vec_t input = vec_size / 4;
  // initialize sketches
  Sketch::configure(vec_size, 100);
  SketchUniquePtr skt = makeSketch(seed);

  // Test the speed of updating the sketches
  for (auto _ : state) {
    ++input;
    skt->update(input);
  }
  state.counters["Updates"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["Hashes"] = benchmark::Counter(
      state.iterations() * (bucket_gen(100) + 1), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Update)->RangeMultiplier(4)->Ranges({{KB << 4, MB << 4}});

// Benchmark the speed of querying sketches
static void BM_Sketch_Query(benchmark::State& state) {
  constexpr size_t vec_size = KB << 5;
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
  state.counters["Query Rate"] = benchmark::Counter(
    state.iterations() * num_sketches, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Query)->DenseRange(0, 90, 10);

// Benchmark speed of DSU merges when the sequence of merges is adversarial
// This means we avoid joining roots wherever possible
static void BM_DSU_Adversarial(benchmark::State &state) {
  constexpr size_t size_of_dsu = 16 * MB;

  auto rng = std::default_random_engine{};

  std::vector<std::pair<node_id_t, node_id_t>> updates;
  // generate updates
  for (size_t iter = 0; ((size_t)2 << iter) <= size_of_dsu; iter++) {
    size_t loc_size = 1 << iter;
    size_t jump = 2 << iter;
    std::vector<std::pair<node_id_t, node_id_t>> new_updates;
    for (size_t i = 0; i < size_of_dsu; i += jump) {
      new_updates.push_back({i + loc_size - 1, i + loc_size - 1 + jump / 2});
    }
    std::shuffle(new_updates.begin(), new_updates.end(), rng);
    updates.insert(updates.end(), new_updates.begin(), new_updates.end());
  }

  // Perform merge test
  for (auto _ : state) {
    DisjointSetUnion<node_id_t> dsu(size_of_dsu);
    for (auto upd : updates) {
      dsu.merge(upd.first, upd.second);
    }
  }
  state.counters["Merge_Latency"] = benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Adversarial);

// Benchmark speed of DSU merges when the sequence of merges is helpful
// this means we only join roots
static void BM_DSU_Root(benchmark::State &state) {
  constexpr size_t size_of_dsu = 16 * MB;

  auto rng = std::default_random_engine{};

  // generate updates
  std::vector<std::pair<node_id_t, node_id_t>> updates;
  // generate updates
  for (size_t iter = 0; ((size_t)2 << iter) <= size_of_dsu; iter++) {
    size_t jump = 2 << iter;
    std::vector<std::pair<node_id_t, node_id_t>> new_updates;
    for (size_t i = 0; i < size_of_dsu; i += jump) {
      new_updates.push_back({i, i + jump / 2});
    }
    std::shuffle(new_updates.begin(), new_updates.end(), rng);
    updates.insert(updates.end(), new_updates.begin(), new_updates.end());
  }

  // Perform merge test
  for (auto _ : state) {
    DisjointSetUnion<node_id_t> dsu(size_of_dsu);
    for (auto upd : updates) {
      dsu.merge(upd.first, upd.second);
    }
  }
  state.counters["Merge_Latency"] = benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Root);

BENCHMARK_MAIN();
