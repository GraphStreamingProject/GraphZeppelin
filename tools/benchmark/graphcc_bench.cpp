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

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

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

static void BM_Multiply(benchmark::State& state) {
  size_t x = 5;
  size_t y = 9;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x = x * y);
    y += 1;
  }
}
BENCHMARK(BM_Multiply);

static void BM_builtin_ffsll(benchmark::State& state) {
  size_t i = 0;
  size_t diff = 1;
  if (state.range(0) == 1) {
    i = size_t(-1);
    diff = -1;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ffsll(i));
    i += diff;
  }
}
BENCHMARK(BM_builtin_ffsll)->DenseRange(0, 1);

static void BM_builtin_ctzll(benchmark::State& state) {
  size_t i = 0;
  size_t diff = 1;
  if (state.range(0) == 1) {
    i = size_t(-1);
    diff = -1;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_ctzll(i));
    i += diff;
  }
}
BENCHMARK(BM_builtin_ctzll)->DenseRange(0, 1);

static void BM_builtin_clzll(benchmark::State& state) {
  size_t i = 0;
  size_t diff = 1;
  if (state.range(0) == 1) {
    i = size_t(-1);
    diff = -1;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(__builtin_clzll(i));
    i += diff;
  }
}
BENCHMARK(BM_builtin_clzll)->DenseRange(0, 1);

// Test the speed of hashing using a method that loops over seeds and a method that
// batches by seed
// The argument to this benchmark is the number of hashes to batch
static void BM_Hash_XXH64(benchmark::State& state) {
  uint64_t input = 100'000;
  size_t seed = get_seed();
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(XXH64(&input, sizeof(uint64_t), seed));
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH64);

static void BM_Hash_XXH3_64(benchmark::State& state) {
  uint64_t input = 100'000;
  size_t seed = get_seed();
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(XXH3_64bits_withSeed(&input, sizeof(uint64_t), seed));
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Hash_XXH3_64);

static void BM_index_depth_hash(benchmark::State& state) {
  uint64_t input = 100'000;
  size_t seed = get_seed();
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(Bucket_Boruvka::get_index_depth(input, seed, 20));
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_index_depth_hash);

static void BM_index_hash(benchmark::State& state) {
  uint64_t input = 100'000;
  size_t seed = get_seed();
  for (auto _ : state) {
    ++input;
    benchmark::DoNotOptimize(Bucket_Boruvka::get_index_hash(input, seed));
  }
  state.counters["Hashes"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_index_hash);

static void BM_update_bucket(benchmark::State& state) {
  Bucket bkt;
  vec_t input = 0x0EADBEEF;
  vec_hash_t checksum = 0x0EEDBEEF;

  for (auto _ : state) {
    ++input;
    ++checksum;
    Bucket_Boruvka::update(bkt, input, checksum);
    benchmark::DoNotOptimize(bkt);
  }
}
BENCHMARK(BM_update_bucket);

// Benchmark the speed of updating sketches both serially and in batch mode
static void BM_Sketch_Update(benchmark::State& state) {
  size_t vec_size = state.range(0);
  vec_t input = vec_size / 3;
  size_t seed = get_seed();
  // initialize sketches
  Sketch skt(vec_size, seed);

  // Test the speed of updating the sketches
  for (auto _ : state) {
    ++input;
    skt.update(input);
  }
  state.counters["Updates"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["Hashes"] =
      benchmark::Counter(state.iterations() * (skt.get_columns() + 1), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Update)->RangeMultiplier(4)->Ranges({{KB << 4, MB << 15}});

// Benchmark the speed of querying sketches
static constexpr size_t sample_vec_size = MB << 10;
static void BM_Sketch_Sample(benchmark::State& state) {
  constexpr size_t num_sketches = 400;

  // initialize sketches with different seeds
  Sketch* sketches[num_sketches];
  for (size_t i = 0; i < num_sketches; i++) {
    sketches[i] = new Sketch(sample_vec_size, get_seed() * 7);
  }

  // perform updates to the sketches (do at least 1)
  for (size_t i = 0; i < num_sketches; i++) {
    for (size_t j = 0; j < size_t(state.range(0)); j++) {
      sketches[i]->update(j + 1);
    }
  }
  SketchSample sample_ret;
  size_t successes = 0;

  for (auto _ : state) {
    // perform queries
    for (size_t j = 0; j < num_sketches; j++) {
      sample_ret = sketches[j]->sample();
      successes += sample_ret.result == GOOD;
      sketches[j]->reset_sample_state();
    }
  }
  state.counters["Samples"] =
      benchmark::Counter(state.iterations() * num_sketches, benchmark::Counter::kIsRate);
  state.counters["Successes"] = double(successes) / (state.iterations() * num_sketches);
}
BENCHMARK(BM_Sketch_Sample)->RangeMultiplier(4)->Range(1, sample_vec_size / 2);

static void BM_Sketch_Merge(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  size_t seed = get_seed();
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
BENCHMARK(BM_Sketch_Merge)->RangeMultiplier(10)->Range(1e3, 1e8);

static void BM_Sketch_Merge_Many(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  size_t seed = get_seed();
  size_t num_sketches = 1 << 7;
  Sketch source(n, seed);
  // Sketch* dests = (Sketch*) malloc(num_sketches * sizeof(Sketch));
  std::vector<Sketch*> dests;
  // TODO - THERES A BUNCH OF UNFREED MEMORY HERE
  for (size_t i=0; i < num_sketches; i++)
    dests.push_back(new Sketch(n, seed));
  // Sketch s2(n, seed);

  for (size_t i = 0; i < upds; i++) {
    source.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
    for (auto dest: dests)
      dest->update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
  }

  for (auto _ : state) {
    for (size_t i=0; i < num_sketches; i++)
      source.merge(*dests[i]);
      // s1.merge(s2);
  }
}
BENCHMARK(BM_Sketch_Merge_Many)->RangeMultiplier(10)->Range(1e3, 1e8);

static void BM_Sketch_Serialize(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  size_t seed = get_seed();
  Sketch s1(n, seed);

  for (size_t i = 0; i < upds; i++) {
    s1.update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
  }

  for (auto _ : state) {
    std::stringstream stream;
    s1.serialize(stream);
  }
}
BENCHMARK(BM_Sketch_Serialize)->RangeMultiplier(10)->Range(1e3, 1e12);

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

// Benchmark DSU Find Root
static void BM_DSU_Find(benchmark::State& state) {
  constexpr size_t size_of_dsu = 16 * MB;
  DisjointSetUnion<node_id_t> dsu(size_of_dsu);

  auto rng = std::default_random_engine{};
  std::vector<node_id_t> queries;
  for (size_t i = 0; i < 4096; i++) {
    queries.push_back((size_of_dsu / 4096) * i);
  }
  std::shuffle(queries.begin(), queries.end(), rng);

  // perform find test
  for (auto _ : state) {
    for (auto q : queries)
      dsu.find_root(q);
  }
  state.counters["Find_Latency"] =
      benchmark::Counter(state.iterations() * queries.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Find);

static void BM_DSU_Find_After_Combine(benchmark::State& state) {
  constexpr size_t size_of_dsu = 16 * MB;
  DisjointSetUnion<node_id_t> dsu(size_of_dsu);
  // merge everything into same root
  for (size_t i = 0; i < size_of_dsu - 1; i++) {
    dsu.merge(i, i+1);
  }

  auto rng = std::default_random_engine{};
  std::vector<node_id_t> queries;
  for (size_t i = 0; i < 4096; i++) {
    queries.push_back((size_of_dsu / 4096) * i);
  }
  std::shuffle(queries.begin(), queries.end(), rng);

  // perform find test
  for (auto _ : state) {
    for (auto q : queries)
      dsu.find_root(q);
  }
  state.counters["Find_Latency"] =
      benchmark::Counter(state.iterations() * queries.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_DSU_Find_After_Combine);

// MT DSU Find Root
static void BM_MT_DSU_Find(benchmark::State& state) {
  constexpr size_t size_of_dsu = 16 * MB;
  DisjointSetUnion_MT<node_id_t> dsu(size_of_dsu);

  auto rng = std::default_random_engine{};
  std::vector<node_id_t> queries;
  for (size_t i = 0; i < 4096; i++) {
    queries.push_back((size_of_dsu / 4096) * i);
  }
  std::shuffle(queries.begin(), queries.end(), rng);

  // perform find test
  for (auto _ : state) {
    for (auto q : queries)
      dsu.find_root(q);
  }
  state.counters["Find_Latency"] =
      benchmark::Counter(state.iterations() * queries.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_MT_DSU_Find);

// MT DSU Find Root
static void BM_MT_DSU_Find_After_Combine(benchmark::State& state) {
  constexpr size_t size_of_dsu = MB;
  DisjointSetUnion_MT<node_id_t> dsu(size_of_dsu);
  // merge everything into same root
  for (size_t i = 0; i < size_of_dsu - 1; i++) {
    dsu.merge(i, i+1);
  }

  auto rng = std::default_random_engine{};
  std::vector<node_id_t> queries;
  for (size_t i = 0; i < 512; i++) {
    queries.push_back((size_of_dsu / 512) * i);
  }
  std::shuffle(queries.begin(), queries.end(), rng);

  // perform find test
  for (auto _ : state) {
    for (auto q : queries)
      dsu.find_root(q);
  }
  state.counters["Find_Latency"] =
      benchmark::Counter(state.iterations() * queries.size(),
                         benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_MT_DSU_Find_After_Combine);

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

#include <cmath>
#include <iterator>
#include <cstddef>

// Fixed-size Hash Set: That is we are given a reasonable upper bound on the size of the set
// so we just allocate the memory we need once and don't worry about resizing.
class FixedSizeHashSet {
 private:
  struct FindResult {
    bool match;
    size_t position;
  };
  size_t max_size;
  size_t cur_size = 0;
  size_t data_slots;
  size_t mask;
  size_t seed;
  bool zero_in_set = false;

  size_t *table; // hash table data, for fast duplicate checking
  size_t *data; // for fast iteration we keep actual data contiguous

  FindResult find_value(size_t value) {
    size_t hash_slot = XXH3_64bits_withSeed(&value, sizeof(value), seed) & mask;

    while (table[hash_slot] != 0 && table[hash_slot] != value) {
      hash_slot = (hash_slot + 1) & mask;
    }

    // is slot empty or does it match value?
    return {table[hash_slot] == value, hash_slot};
  }
 public:
  // Iterator for accessing the contents of the set
  struct Iterator {
   private:
    const size_t *ptr;

   public:
    // iterator tags
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = const size_t;
    using pointer = const size_t *;
    using reference = const size_t &;

    Iterator(size_t *ptr) : ptr(ptr) {}

    // access contents
    reference operator*() const { return *ptr; }
    pointer operator->() { return ptr; }

    // Increment the iterator
    Iterator& operator++() { 
      ptr++;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // comparison operators
    friend bool operator== (const Iterator& a, const Iterator& b) { return a.ptr == b.ptr; };
    friend bool operator!= (const Iterator& a, const Iterator& b) { return a.ptr != b.ptr; };
  };

  // Iterator begin and end
  Iterator begin() {
    return Iterator(&data[0]);
  }
  Iterator end() {
    return Iterator(&data[cur_size]);
  }

  FixedSizeHashSet(size_t max_set_size, size_t seed)
      : max_size(max_set_size),
        data_slots(size_t(1) << size_t(ceil(log2(max_size)))), // TODO: augment size
        mask(data_slots - 1),
        seed(seed),
        table(new size_t[data_slots]),
        data(new size_t[max_size]) {
    for (size_t i = 0; i < data_slots; i++) {
      table[i] = 0;
    }
    for (size_t i = 0; i < max_size; i++) {
      data[i] = 0;
    }
  }
  ~FixedSizeHashSet() {
    delete[] table;
    delete[] data;
  }

  // add an element x to the set
  // returns true if successful
  // returns false if x is already a member of the set
  bool insert(size_t x) {
    unlikely_if (x == 0) {
      if (zero_in_set) return false;
      zero_in_set = true;
      data[cur_size++] = 0;
      return true;
    }

    FindResult m = find_value(x);
    if (m.match) return false;
    
    // add new element to set
    table[m.position] = x;
    data[cur_size++] = x;
    return true;
  }
  
  // check if a given element is in the set
  // returns true if x is member of set, false otherwise
  bool check(size_t x) {
    unlikely_if (x == 0) return zero_in_set;

    return find_value(x).match;
  }

  // clear all entries from the hash set
  void clear() {
    for (size_t i = 0; i < data_slots; i++) {
      table[i] = 0;
    }
    for (size_t i = 0; i < max_size; i++) {
      data[i] = 0;
    }
    zero_in_set = false;
  }
};

static void BM_Fixed_Size_Hash_Insert(benchmark::State& state) {
  constexpr size_t size = 1e6;
  size_t seed = get_seed();
  FixedSizeHashSet set(size, seed);

  size_t x = 0;
  size_t range = size / state.range(0);
  for (auto _ : state) {
    set.insert(x);
    x = ((x + 1) % range);
  }

  state.counters["Insert_Latency"] = benchmark::Counter(
      state.iterations(), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Fixed_Size_Hash_Insert)->RangeMultiplier(2)->Range(1, 1 << 14);

static void BM_Std_Set_Hash_Insert(benchmark::State& state) {
  constexpr size_t size = 1e6;
  std::unordered_set<size_t> set;

  size_t x = 0;
  size_t range = size / state.range(0);
  for (auto _ : state) {
    set.insert(x);
    x = ((x + 1) % range);
  }

  state.counters["Insert_Latency"] = benchmark::Counter(
      state.iterations(), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Std_Set_Hash_Insert)->RangeMultiplier(2)->Range(1, 1 << 14);

static void BM_Fixed_Size_Hash_Iterator(benchmark::State& state) {
  constexpr size_t size = 1e6;
  size_t seed = get_seed();
  FixedSizeHashSet set(size, seed);

  size_t range = size / state.range(0);
  for (size_t i = 0; i < range; i++) {
    set.insert(i);
  }

  for (auto _ : state) {
    // iterate over the set
    for (auto &elm : set) {
      benchmark::DoNotOptimize(elm);
    }
  }

  state.counters["Scan_Latency"] = benchmark::Counter(
      state.iterations(), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Fixed_Size_Hash_Iterator)->RangeMultiplier(2)->Range(1, 1 << 14);

static void BM_Std_Set_Hash_Iterator(benchmark::State& state) {
  constexpr size_t size = 1e6;
  std::unordered_set<size_t> set;

  size_t range = size / state.range(0);
  for (size_t i = 0; i < range; i++) {
    set.insert(i);
  }

  for (auto _ : state) {
    // iterate over the set
    for (auto &elm : set) {
      benchmark::DoNotOptimize(elm);
    }
  }

  state.counters["Scan_Latency"] = benchmark::Counter(
      state.iterations(), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Std_Set_Hash_Iterator)->RangeMultiplier(2)->Range(1, 1 << 14);

BENCHMARK_MAIN();