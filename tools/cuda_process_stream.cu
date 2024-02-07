#include <vector>
#include <graph_sketch_driver.h>
#include <cc_gpu_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include <sys/resource.h> // for rusage
#include <cuda_kernel.cuh>

static bool shutdown = false;

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void track_insertions(uint64_t total, GraphSketchDriver<CCGPUSketchAlg> *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2; // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev  = start_time;
  uint64_t prev_updates = 0;

  while(true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff   = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown)
      break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(), std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec); fflush(stdout);
  }

  printf("Progress:====================| Done                             \n");
  return;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  int reader_threads = std::atoi(argv[3]);

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  auto cc_config = CCAlgConfiguration().batch_factor(1);
  CCGPUSketchAlg cc_gpu_alg{num_nodes, get_seed(), cc_config};
  GraphSketchDriver<CCGPUSketchAlg> driver{&cc_gpu_alg, &stream, driver_config, reader_threads};
  
  Sketch** sketches = cc_gpu_alg.get_sketches();
  
  // Get variables from sketch
  size_t num_samples = sketches[0]->get_num_samples();
  size_t num_buckets = sketches[0]->get_buckets();
  size_t num_columns = sketches[0]->get_columns();
  size_t bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));

  std::cout << "num_samples: " << num_samples << "\n"; // num_sketches = num_samples
  std::cout << "num_buckets: " << num_buckets << "\n"; // num_elems = num_buckets
  std::cout << "num_columns: " << num_columns << "\n";
  std::cout << "bkt_per_col: " << bkt_per_col << "\n"; // num_guesses = bkt_per_col

  // Start timer for initializing
  auto init_start = std::chrono::steady_clock::now();

  GutteringSystem *gts = driver.get_gts();
  int batch_size = gts->gutter_size() / sizeof(node_id_t);
  int stream_multiplier = 4;

  std::cout << "Batch_size: " << batch_size << "\n";
  
  CudaUpdateParams* cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
  cudaUpdateParams = new CudaUpdateParams(num_nodes, num_updates, num_samples, num_buckets, num_columns, bkt_per_col, num_threads, batch_size, stream_multiplier);

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_nodes * sizeof(long)));

  for (node_id_t i = 0; i < num_nodes; i++) {
    sketchSeeds[i] = sketches[i]->get_seed();
  }

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";

  size_t maxBytes = num_buckets * sizeof(vec_t_cu) + num_buckets * sizeof(vec_hash_t);
  cc_gpu_alg.updateKernelSharedMemory(maxBytes);
  std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

  // Prefetch memory to device 
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, num_nodes * sizeof(long), device_id));
  cc_gpu_alg.configure(&cudaUpdateParams, sketchSeeds, num_threads);
  
  std::cout << "Finished initializing CUDA parameters\n";
  std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
  std::cout << "CUDA parameters init duration: " << init_time.count() << std::endl;

  // Start timer for kernel
  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  // Call kernel code
  std::cout << "Update Kernel Starting...\n";

  // Perform edge updates
  driver.process_stream_until(END_OF_STREAM);

  std::cout << "Update Kernel finished.\n";

  // Start timer for cc
  auto cc_start = std::chrono::steady_clock::now();
  
  std::cout << "  Flush Starting...\n";

  auto flush_start = std::chrono::steady_clock::now();

  gts->force_flush();
  driver.get_worker_threads()->flush_workers();
  cudaDeviceSynchronize();
  cc_gpu_alg.apply_flush_updates();

  auto flush_end = std::chrono::steady_clock::now();

  /*for (int i = 0; i < num_nodes; i++){
    //std::cout << cudaUpdateParams[0].h_bucket_a[i] << " ";
    int count = 0;
    for (int j = 0; j < num_buckets; j++) {
      if (sketches[i]->get_readonly_bucket_ptr()[j].alpha != 0) {
        count++;
      } 
    }
    std::cout << "Node " << i << ": " << count << "\n";
    
  }*/

  std::cout << "  Flushed Ended.\n";

  auto CC_num = cc_gpu_alg.connected_components().size();

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> cc_alg_time = cc_gpu_alg.cc_alg_end - cc_gpu_alg.cc_alg_start;

  shutdown = true;
  querier.join();

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec):    " << num_seconds << std::endl;
  std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;
  std::cout << "Total CC query latency:       " << cc_time.count() + flush_time.count() << std::endl;
  std::cout << "  Flush Gutters(sec):           " << flush_time.count() << std::endl;
  std::cout << "  Boruvka's Algorithm(sec):     " << cc_alg_time.count() << std::endl;
  std::cout << "Connected Components:         " << CC_num << std::endl;
  std::cout << "Maximum Memory Usage(MiB):    " << get_max_mem_used() << std::endl;
}
