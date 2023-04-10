#include <vector>
#include <graph.h>
#include <map>
#include <binary_graph_stream.h>
#include "../src/cuda_kernel.cu"

typedef std::pair<Edge, UpdateType> GraphUpdate;

// CUDA API Check
// Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
  else if (num_threads_per_update == 4) {
    std::cout << "Running with four threads for each edge update" << std::endl;
  }
  else {
    std::cout << "ERROR: Invalid number of threads per edge update. Must be 1, 2, or 4." << std::endl;
  }

  BinaryGraphStream stream(stream_file, 1024*32);
  node_id_t num_nodes = stream.nodes();
  size_t num_updates  = stream.edges();
  std::cout << "Running process_stream with CUDA: " << std::endl;
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto config = GraphConfiguration().gutter_sys(STANDALONE).num_groups(num_threads);
  config.gutter_conf().gutter_factor(-4);
  Graph g{num_nodes, config};

  int *nodeUpdates;
  gpuErrchk(cudaMallocManaged(&nodeUpdates, num_updates * sizeof(int) * 2));

  vec_t *edgeUpdates;
  gpuErrchk(cudaMallocManaged(&edgeUpdates, num_updates * sizeof(vec_t) * 2));

  Supernode** supernodes;
  supernodes = g.getSupernodes();

  std::map<int, std::vector<vec_t>> graphUpdates;

  for (int i = 0; i < num_nodes; i++) {
    graphUpdates[i] = std::vector<vec_t>{};
  }
  
  // Collect all the edges that need to be updated
  // 1 Thread will be assigned to update the endpoint nodes of each edge
  for (size_t e = 0; e < num_updates; e++) {
    GraphUpdate graphUpdate = stream.get_edge();
    Edge updatedEdge = graphUpdate.first;
    
    graphUpdates[updatedEdge.first].push_back(static_cast<vec_t>(concat_pairing_fn(updatedEdge.first, updatedEdge.second)));
    graphUpdates[updatedEdge.second].push_back(static_cast<vec_t>(concat_pairing_fn(updatedEdge.second, updatedEdge.first)));   
  }

  std::cout << "Finished initializing graphUpdates\n";

  int nodeIt = 0;
  for (auto it = graphUpdates.begin(); it != graphUpdates.end(); it++) {
    for (int i = 0; i < it->second.size(); i++) {
      nodeUpdates[nodeIt] = it->first;
      edgeUpdates[nodeIt] = it->second.at(i);
      nodeIt++;
    }
  }

  std::cout << "Finished initializing nodeUpdates and edgeUpdates\n";

  // Previous version of inserting nodeUpdates and edgeUpdates.
  /*
  for (size_t e = 0; e < num_updates; e++) {
    GraphUpdate graphUpdate = stream.get_edge();
    Edge updatedEdge = graphUpdate.first;
    nodeUpdates[(e * 2)] = updatedEdge.first;
    nodeUpdates[(e * 2) + 1] = updatedEdge.second;
    edgeUpdates[(e * 2)] = static_cast<vec_t>(concat_pairing_fn(updatedEdge.first, updatedEdge.second));
    edgeUpdates[(e * 2) + 1] = static_cast<vec_t>(concat_pairing_fn(updatedEdge.second, updatedEdge.first));
  }*/

  // Get number of sketches for each node.
  // Number of sketches stays constant for among all super nodes.
  int num_sketches = supernodes[0]->get_num_sktch();

  std::cout << "num_sketches: " << num_sketches << "\n";

  CudaSketch* cudaSketches;
  gpuErrchk(cudaMallocManaged(&cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch)));

  // Allocate space for all buckets

  int num_elems = supernodes[0]->get_sketch(0)->get_num_elems();
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

      sketch->set_bucket_a(bucket_a);
      sketch->set_bucket_c(bucket_c);

      CudaSketch cudaSketch(bucket_a, bucket_c, sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed());
      cudaSketches[(i * num_sketches) + j] = cudaSketch;
    }
  }

  // Number of threads
  int num_device_threads = 1024;
  
  // Number of blocks
  int num_device_blocks = 1;

  if(num_threads_per_update == 1) {
    num_device_blocks = (num_updates + num_device_threads - 1) / num_device_threads;
  }
  else if(num_threads_per_update == 2){ // Need twice number of total threads in grid
    num_device_blocks = ((num_updates * 2) + num_device_threads - 1) / num_device_threads;
  }
  else { // Need four times more number of total threads in grid 
    num_device_blocks = ((num_updates * 4) + num_device_threads - 1) / num_device_threads;
  }

  int device_id = cudaGetDevice(&device_id);

  // Prefetch memory to device 
  gpuErrchk(cudaMemPrefetchAsync(nodeUpdates, num_updates * sizeof(int) * 2, device_id));
  gpuErrchk(cudaMemPrefetchAsync(edgeUpdates, num_updates * sizeof(vec_t) * 2, device_id));
  gpuErrchk(cudaMemPrefetchAsync(cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_a, num_nodes * num_sketches * num_elems * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_c, num_nodes * num_sketches * num_elems * sizeof(vec_hash_t), device_id));

  auto ins_start = std::chrono::steady_clock::now();

  // Call kernel code
  std::cout << "Kernel Starting...\n";
  streamUpdate(num_device_threads, num_device_blocks, nodeUpdates, num_updates, num_nodes, num_sketches, num_elems, edgeUpdates, 
              cudaSketches, num_threads_per_update);
  std::cout << "Kernel finished.\n";

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
