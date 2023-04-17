#include <vector>
#include <graph.h>
#include <map>
#include <binary_graph_stream.h>
#include "../src/cuda_kernel.cu"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, number of threads per edge update" << std::endl;
  }

  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
  }

  int num_threads_per_update = std::atoi(argv[3]);
  if (num_threads_per_update == 1) {
    std::cout << "Running with one thread for each edge update" << std::endl;
  }
  else if (num_threads_per_update == 2) {
    std::cout << "Running with two threads for each edge update" << std::endl;
  }
  else {
    std::cout << "ERROR: Invalid number of threads per edge update. Must be 1 or 2." << std::endl;
  }

  BinaryGraphStream_MT stream(stream_file, 1024*32);
  node_id_t num_nodes = stream.nodes();
  size_t num_updates  = stream.edges();
  std::cout << "Running process_stream with CUDA: " << std::endl;
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto config = GraphConfiguration().gutter_sys(STANDALONE).num_groups(num_threads);
  config.gutter_conf().gutter_factor(-4);
  Graph g{num_nodes, config, 1};

  Supernode** supernodes;
  supernodes = g.getSupernodes();

  // Get variable from sample supernode
  int num_sketches = supernodes[0]->get_num_sktch();
  
  // Get variables from sample sketch
  size_t num_elems = supernodes[0]->get_sketch(0)->get_num_elems();
  size_t num_columns = supernodes[0]->get_sketch(0)->get_columns();
  size_t num_guesses = supernodes[0]->get_sketch(0)->get_num_guesses();
  
  CudaParams* cudaParams;
  gpuErrchk(cudaMallocManaged(&cudaParams, sizeof(CudaParams)));
  cudaParams[0] = CudaParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses);

  // Hashmap that stores node ids and edge ids that need to be updated
  std::map<int, std::vector<vec_t>> graphUpdates;

  // Initialize with empty vector
  for (int i = 0; i < num_nodes; i++) {
    graphUpdates[i] = std::vector<vec_t>{};
    cudaParams[0].nodeNumUpdates[i] = 0;
    cudaParams[0].nodeStartIndex[i] = 0;
  }
  
  MT_StreamReader reader(stream);
  GraphUpdate upd;

  // Collect all the edges that need to be updated
  for (size_t e = 0; e < num_updates; e++) {
    upd = reader.get_edge();
    Edge &edge = upd.edge;

    graphUpdates[edge.src].push_back(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
    graphUpdates[edge.dst].push_back(static_cast<vec_t>(concat_pairing_fn(edge.dst, edge.src)));   
  }

  std::cout << "Finished initializing graphUpdates\n";

  // Transfer graphUpdates to nodeUpdates and edgeUpdates
  int nodeIt = 0;
  int startIndex = 0;
  for (auto it = graphUpdates.begin(); it != graphUpdates.end(); it++) {
    cudaParams[0].nodeStartIndex[it->first] = startIndex;
    cudaParams[0].nodeNumUpdates[it->first] = it->second.size();
    for (int i = 0; i < it->second.size(); i++) {
      cudaParams[0].nodeUpdates[nodeIt] = it->first;
      cudaParams[0].edgeUpdates[nodeIt] = it->second.at(i);
      nodeIt++;
    }
    startIndex += it->second.size();
  }

  std::cout << "Finished initializing nodeUpdates and edgeUpdates\n";

  CudaSketch* cudaSketches;
  gpuErrchk(cudaMallocManaged(&cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch)));

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_nodes * num_sketches * sizeof(long)));

  // Allocate space for all buckets
  vec_t* d_bucket_a;
  vec_hash_t* d_bucket_c;
  gpuErrchk(cudaMallocManaged(&d_bucket_a, (num_nodes * num_sketches * num_elems * sizeof(vec_t))));
  gpuErrchk(cudaMallocManaged(&d_bucket_c, (num_nodes * num_sketches * num_elems * sizeof(vec_hash_t))));

  for (int i = 0; i < (num_nodes * num_sketches * num_elems); i++) {
    d_bucket_a[i] = 0;
    d_bucket_c[i] = 0;
  }

  // Create a vector of cuda supernodes and sketches
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_sketches; j++) {
      Sketch* sketch = supernodes[i]->get_sketch(j);

      int bucket_id = (i * num_sketches * num_elems) + (j * num_elems);
      vec_t* bucket_a = &d_bucket_a[bucket_id];
      vec_hash_t* bucket_c = &d_bucket_c[bucket_id];

      // Rewrite sketch's bucket_a and bucket_c memory location
      sketch->set_bucket_a(bucket_a);
      sketch->set_bucket_c(bucket_c);

      CudaSketch cudaSketch(bucket_a, bucket_c, sketch->get_seed());
      cudaSketches[(i * num_sketches) + j] = cudaSketch;
      sketchSeeds[(i * num_sketches) + j] = sketch->get_seed();
    }
  }

  // Number of threads
  int num_device_threads = 1024;
  
  // Number of blocks
  int num_device_blocks = 0;

  if(num_threads_per_update == 1) {
    num_device_blocks = (num_updates + num_device_threads - 1) / num_device_threads;
  }
  else { // Need twice number of total threads in grid
    //num_device_blocks = ((num_updates * 2) + num_device_threads - 1) / num_device_threads;
    num_device_blocks = num_nodes;
  }

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";

  std::cout << "Allocated Shared Memory of: " << (num_elems * num_sketches * sizeof(vec_t_cu)) + (num_elems * num_sketches * sizeof(vec_hash_t)) << "\n";

  // Prefetch memory to device 
  gpuErrchk(cudaMemPrefetchAsync(cudaParams, sizeof(CudaParams), device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaParams[0].nodeUpdates, num_updates * sizeof(node_id_t) * 2, device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaParams[0].edgeUpdates, num_updates * sizeof(vec_t) * 2, device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaParams[0].nodeNumUpdates, num_nodes * sizeof(node_id_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaParams[0].nodeStartIndex, num_nodes * sizeof(node_id_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch), device_id));
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, num_nodes * num_sketches * sizeof(long), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_a, num_nodes * num_sketches * num_elems * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_c, num_nodes * num_sketches * num_elems * sizeof(vec_hash_t), device_id));

  // Start timer for kernel
  auto ins_start = std::chrono::steady_clock::now();

  // Call kernel code
  std::cout << "Kernel Starting...\n";
  streamUpdate(num_device_threads, num_device_blocks, cudaParams, cudaSketches, sketchSeeds, num_threads_per_update);
  std::cout << "Kernel finished.\n";

  // End timer for kernel
  auto cc_start = std::chrono::steady_clock::now();
  
  // Update graph's num_updates value
  g.num_updates += num_updates * 2;

  // Prefetch bucket memory back to CPU
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_a, num_nodes * num_sketches * num_elems * sizeof(vec_t), cudaCpuDeviceId));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_c, num_nodes * num_sketches * num_elems * sizeof(vec_hash_t), cudaCpuDeviceId));

  auto CC_num = g.connected_components().size();
  std::chrono::duration<double> insert_time = cc_start - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  double num_seconds = insert_time.count();
  std::cout << "Total insertion time was: " << num_seconds << std::endl;
  std::cout << "Insertion rate was:       " << stream.edges() / num_seconds << std::endl;
  std::cout << "CC query latency:         " << cc_time.count() << std::endl;
  std::cout << "Connected Components:     " << CC_num << std::endl;
}
