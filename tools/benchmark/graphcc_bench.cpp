#include <benchmark/benchmark.h>
#include <unistd.h>
#include <xxhash.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <sstream>

#include "binary_file_stream.h"
#include "bucket.h"
#include "dsu.h"
#include "sketch.h"

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
    BinaryFileStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary");
    num_edges = stream.edges();
  }

  // flush fs cache
  flush_filesystem_cache();

  // perform benchmark
  for (auto _ : state) {
    BinaryFileStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary");

    bool reading = true;
    while (reading) {
      GraphStreamUpdate upds[state.range(0)];
      size_t num_updates = stream->get_update_buffer(upds, state.range(0));
      for (size_t i = 0; i < num_updates; i++) {
        GraphStreamUpdate &upd = upds[i];
        if (upd.type == BREAKPOINT) {
          reading = false;
          break;
        }
      }
    }
  }
  state.counters["Ingestion_Rate"] =
      benchmark::Counter(state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_FileIngest)->RangeMultiplier(2)->Range(KB << 2, MB / 4)->UseRealTime();

// Test the speed of reading all the data in the kron16 graph stream
static void BM_MTFileIngest(benchmark::State& state) {
  // determine the number of edges in the graph
  uint64_t num_edges;
  {
    BinaryFileStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary");
    num_edges = stream.edges();
  }

  // flush fs cache
  flush_filesystem_cache();

  // perform benchmark
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(state.range(0));

    BinaryFileStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary");

    auto task = [&]() {
      bool reading = true;
      while (reading) {
        GraphStreamUpdate upds[1024];
        size_t num_updates = stream->get_update_buffer(upds, 1024);
        for (size_t i = 0; i < num_updates; i++) {
          GraphStreamUpdate &upd = upds[i];
          if (upd.type == BREAKPOINT) {
            reading = false;
            break;
          }
        }
      }
    };

    for (int i = 0; i < state.range(0); i++) threads.emplace_back(task);
    for (int i = 0; i < state.range(0); i++) threads[i].join();
  }
  state.counters["Ingestion_Rate"] =
      benchmark::Counter(state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_MTFileIngest)->RangeMultiplier(4)->Range(1, 20)->UseRealTime();
#endif  // FILE_INGEST_F

static void BM_builtin_ffsll(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ffsll(i++));
    benchmark::DoNotOptimize(__builtin_ffsll(j--));
  }
}
BENCHMARK(BM_builtin_ffsll);

static void BM_builtin_ctzll(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ctzll(i++));
    benchmark::DoNotOptimize(__builtin_ctzll(j--));
  }
}
BENCHMARK(BM_builtin_ctzll);

static void BM_builtin_clzll(benchmark::State& state) {
  size_t i = 0;
  size_t j = -1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_clzll(i++));
    benchmark::DoNotOptimize(__builtin_clzll(j--));
  }
}
BENCHMARK(BM_builtin_clzll);

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
  vec_t input = vec_size / 3;
  // initialize sketches
  Sketch skt(vec_size, seed, 1, Sketch::default_cols_per_sample);

  // Test the speed of updating the sketches
  for (auto _ : state) {
    ++input;
    skt.update(input);
  }
  state.counters["Updates"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["Hashes"] =
      benchmark::Counter(state.iterations() * (skt.get_columns() + 1), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Update)->RangeMultiplier(4)->Ranges({{KB << 4, MB << 4}});

// Benchmark the speed of querying sketches
static void BM_Sketch_Query(benchmark::State& state) {
  constexpr size_t vec_size = KB << 5;
  constexpr size_t num_sketches = 100;
  double density = ((double)state.range(0)) / 100;

  // initialize sketches
  Sketch* sketches[num_sketches];
  for (size_t i = 0; i < num_sketches; i++) {
    sketches[i] = new Sketch(vec_size, seed, 1, Sketch::default_cols_per_sample);
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
      benchmark::DoNotOptimize(q_ret = sketches[j]->sample());
      sketches[j]->reset_sample_state();
    }
  }
  state.counters["Query Rate"] =
      benchmark::Counter(state.iterations() * num_sketches, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Query)->DenseRange(0, 90, 10);

static void BM_Sketch_Merge(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  Sketch s1(n, seed);
  Sketch s2(n, seed);

  for (size_t i = 0; i < upds; i++) {
    s1.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
    s2.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
  }

  for (auto _ : state) {
    s1.merge(s2);
  }
}
BENCHMARK(BM_Sketch_Merge)->RangeMultiplier(10)->Range(1e3, 1e6);

static void BM_Sketch_Serialize(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  Sketch s1(n, seed);

  for (size_t i = 0; i < upds; i++) {
    s1.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
  }

  for (auto _ : state) {
    std::stringstream stream;
    s1.serialize(stream);
  }
}
BENCHMARK(BM_Sketch_Serialize)->RangeMultiplier(10)->Range(1e3, 1e6);

// static void BM_Sketch_Sparse_Serialize(benchmark::State& state) {
//   size_t n = state.range(0);
//   size_t upds = n / 100;
//   Sketch s1(n, seed);

//   for (size_t i = 0; i < upds; i++) {
//     s1.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
//   }

//   for (auto _ : state) {
//     std::stringstream stream;
//     s1.serialize(stream, SPARSE);
//   }
// }
// BENCHMARK(BM_Sketch_Sparse_Serialize)->RangeMultiplier(10)->Range(1e3, 1e6);

// Benchmark speed of DSU merges when the sequence of merges is adversarial
// This means we avoid joining roots wherever possible
static void BM_DSU_Adversarial(benchmark::State& state) {
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
  state.counters["Merge_Latency"] =
      benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Adversarial);

// Benchmark speed of DSU merges when the sequence of merges is helpful
// this means we only join roots
static void BM_DSU_Root(benchmark::State& state) {
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
  state.counters["Merge_Latency"] =
      benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Root);

// Benchmark the efficiency of parallel DSU merges
// when the sequence of DSU merges is adversarial
// This means we avoid joining roots wherever possible
static void BM_Parallel_DSU_Adversarial(benchmark::State& state) {
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
    DisjointSetUnion_MT<node_id_t> dsu(size_of_dsu);
#pragma omp parallel for num_threads(state.range(0))
    for (auto upd : updates) {
      dsu.merge(upd.first, upd.second);
    }
  }
  state.counters["Merge_Latency"] =
      benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Parallel_DSU_Adversarial)->RangeMultiplier(2)->Range(1, 8)->UseRealTime();

// Benchmark the efficiency of parallel DSU merges
// when the sequence of DSU merges is helpful
// this means we only join roots
static void BM_Parallel_DSU_Root(benchmark::State& state) {
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
    DisjointSetUnion_MT<node_id_t> dsu(size_of_dsu);
#pragma omp parallel for num_threads(state.range(0))
    for (auto upd : updates) {
      dsu.merge(upd.first, upd.second);
    }
  }
  state.counters["Merge_Latency"] =
      benchmark::Counter(state.iterations() * updates.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Parallel_DSU_Root)->RangeMultiplier(2)->Range(1, 8)->UseRealTime();

BENCHMARK_MAIN();
