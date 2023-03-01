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
  state.counters["Ingestion_Rate"] =
      benchmark::Counter(state.iterations() * num_edges, benchmark::Counter::kIsRate);
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
  state.counters["Ingestion_Rate"] =
      benchmark::Counter(state.iterations() * num_edges, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_MTFileIngest)->RangeMultiplier(4)->Range(1, 20)->UseRealTime();
#endif  // FILE_INGEST_F

#include <math.h>

#include <atomic>
#include <exception>
#include <random>
#include <string>
#include <unordered_map>

#include "types.h"

#pragma pack(push, 1)
struct GraphStreamUpdate {
  uint8_t type;
  Edge edge;
};
#pragma pack(pop)

static constexpr edge_id_t END_OF_STREAM = (edge_id_t)-1;

// Enum that defines the types of streams
enum StreamType {
  BinaryFile,
  AsciiFile,
  ErdosStream,
};

// TODO: eventually make this a file in the GraphZeppelin repo
class GraphStream {
 public:
  virtual ~GraphStream() = default;
  inline node_id_t vertices() { return num_vertices; }
  inline edge_id_t edges() { return num_edges; }

  // Extract a buffer of many updates from the stream
  virtual size_t get_update_buffer(GraphStreamUpdate* upd_buf, edge_id_t num_updates) = 0;

  // Query the GraphStream to see if get_update_buffer is thread-safe
  // this is implemenation dependent
  virtual bool get_update_is_thread_safe() = 0;

  // Move read pointer to new location in stream
  // Child classes may choose to throw an error if seek is called
  // For example, a GraphStream recieved over the network would
  // likely not support seek
  virtual void seek(edge_id_t edge_idx) = 0;

  // Query handling
  // Call this function to register a query at a future edge index
  // This function returns true if the query is correctly registered
  virtual bool set_break_point(edge_id_t query_idx) = 0;

  // Serialize GraphStream metadata for distribution
  // So that stream reading can happen simultaneously
  virtual void serialize_metadata(std::ostream& out) = 0;

  // construct a stream object from serialized metadata
  static GraphStream* construct_stream_from_metadata(std::istream& in);

 protected:
  node_id_t num_vertices = 0;
  edge_id_t num_edges = 0;

 private:
  static std::unordered_map<size_t, GraphStream* (*)(std::istream&)> constructor_map;
};

class StreamException : public std::exception {
 private:
  std::string err_msg;

 public:
  StreamException(std::string err) : err_msg(err) {}
  virtual const char* what() const throw() { return err_msg.c_str(); }
};

class ErdosRenyiStream : public GraphStream {
 public:
  ErdosRenyiStream(size_t seed, node_id_t nodes, double density)
      : seed(seed), desired_density(density), source_states(new SrcState[nodes - 1]) {
    node_id_t temp_verts = 1 << (size_t)ceil(log2(nodes));  // round up to nearest power of 2
    if (temp_verts != nodes) {
      std::cerr << "WARNING: Rounding up number of vertices: " << nodes << " to nearest power of 2"
                << std::endl;
    }
    num_vertices = temp_verts;
    vertices_mask = num_vertices - 1;
    upd_idx = 0;
    breakpoint_idx = END_OF_STREAM;

    std::mt19937_64 gen(seed);
    for (size_t i = 0; i < num_vertices - 1; i++) {
      source_states[i].ins_idx = gen() & get_dst_mask(i + 1);
      source_states[i].del_idx = source_states[i].ins_idx;
    }
  }

  inline size_t get_update_buffer(GraphStreamUpdate* upd_buf, size_t num_updates) {
    assert(upd_buf != nullptr);
    edge_id_t local_idx = upd_idx.fetch_add(num_updates, std::memory_order_relaxed);
    if (local_idx + num_updates > breakpoint_idx) {
      upd_idx = breakpoint_idx.load();
      num_updates = local_idx > breakpoint_idx ? 0 : breakpoint_idx - local_idx;
      upd_buf[num_updates] = {BREAKPOINT, {0, 0}};
    }

    // populate the update buffer
    for (size_t i = 0; i < num_updates; i++) upd_buf[i] = create_update(local_idx++);
    return num_updates;
  }

  inline bool get_update_is_thread_safe() { return false; }

  inline void seek(edge_id_t) { throw StreamException("ErdosRenyiStream: seek() not supported!"); }

  inline bool set_break_point(edge_id_t break_idx) {
    if (break_idx < upd_idx) return false;
    breakpoint_idx = break_idx;
    return true;
  }

  inline void serialize_metadata(std::ostream& out) {
    out << ErdosStream << " " << seed << desired_density << std::endl;
  }

 private:
  struct SrcState {
    node_id_t ins_idx;
    node_id_t del_idx;
  };

  inline node_id_t get_dst_mask(node_id_t first_dst) {
    return ~(node_id_t(-1) << (node_id_bits - __builtin_clzl(first_dst)));
  }

  inline node_id_t incr_idx(node_id_t idx, node_id_t amt, node_id_t src) {
    return (idx + amt) & get_dst_mask(src + 1);
  }

  inline node_id_t decr_idx(node_id_t idx, node_id_t amt, node_id_t src) {
    return (idx - amt) & get_dst_mask(src + 1);
  }

  // some basic stream parameters
  const size_t seed;
  const double desired_density;
  node_id_t vertices_mask;
  static constexpr size_t random_range_size = 16;
  static constexpr col_hash_t range_set_mask = (col_hash_t)-1 << (size_t)log2(random_range_size);
  static constexpr col_hash_t node_id_bits = sizeof(node_id_t) * 8;
  static constexpr col_hash_t node_id_mask = (1ull << node_id_bits) - 1;

  // current state of the stream
  bool delete_mode = false;
  std::atomic<edge_id_t> upd_idx;
  std::atomic<edge_id_t> breakpoint_idx;
  std::unique_ptr<SrcState[]> source_states;

  // Helper functions
  inline GraphStreamUpdate create_update(edge_id_t idx) {
    node_id_t src = get_random_src(idx);

    bool del = source_states[src].ins_idx == source_states[src].del_idx || delete_mode;
    del = del && incr_idx(source_states[src].ins_idx, 1, src + 1) == source_states[src].del_idx;
    node_id_t cur_idx = del * source_states[src].del_idx + !del * source_states[src].ins_idx;
    node_id_t dst = get_dst(src, cur_idx);

    source_states[src].ins_idx = incr_idx(source_states[src].ins_idx, !del, src);
    source_states[src].del_idx = incr_idx(source_states[src].del_idx, del, src);

    return {del, {src, dst}};
  }

  inline node_id_t get_random_src(size_t update_idx) {
    col_hash_t hash = col_hash(&update_idx, sizeof(update_idx), seed);
    node_id_t even_hash = (hash >> node_id_bits) & (vertices_mask - 1);
    node_id_t odd_hash = ((hash & node_id_mask) & vertices_mask) | 1;
    return std::min(even_hash, odd_hash);
  }

  inline node_id_t get_dst(node_id_t src, node_id_t idx) { return src + idx; }
};

static void BM_erdos_renyi_stream(benchmark::State& state) {
  node_id_t num_vertices = state.range(0);
  double density = 0.15;
  ErdosRenyiStream stream(seed, num_vertices, density);
  size_t buffer_size = 2048;
  for (auto _ : state) {
    GraphStreamUpdate upd[buffer_size];
    stream.get_update_buffer(upd, buffer_size);
  }
  state.counters["Updates"] =
      benchmark::Counter(state.iterations() * buffer_size, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_erdos_renyi_stream)->RangeMultiplier(2)->Ranges({{1 << 15, 1 << 20}});

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
  Sketch::configure(vec_size, 100);
  SketchUniquePtr skt = makeSketch(seed);

  // Test the speed of updating the sketches
  for (auto _ : state) {
    ++input;
    skt->update(input);
  }
  state.counters["Updates"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["Hashes"] =
      benchmark::Counter(state.iterations() * (bucket_gen(100) + 1), benchmark::Counter::kIsRate);
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
  state.counters["Query Rate"] =
      benchmark::Counter(state.iterations() * num_sketches, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Sketch_Query)->DenseRange(0, 90, 10);

static void BM_Supernode_Merge(benchmark::State& state) {
  size_t n = state.range(0);
  size_t upds = n / 100;
  Supernode::configure(n);
  Supernode* s1 = Supernode::makeSupernode(n, seed);
  Supernode* s2 = Supernode::makeSupernode(n, seed);

  for (size_t i = 0; i < upds; i++) {
    s1->update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
    s2->update(static_cast<vec_t>(concat_pairing_fn(rand() % n, rand() % n)));
  }

  for (auto _ : state) {
    s1->merge(*s2);
  }

  free(s1);
  free(s2);
}
BENCHMARK(BM_Supernode_Merge)->RangeMultiplier(10)->Range(1e3, 1e6);

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

BENCHMARK_MAIN();
