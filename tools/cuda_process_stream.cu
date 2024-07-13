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

  // Get variables from sketch
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "num_samples: " << sketchParams.num_samples << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n"; 

  // Allocate memory for buckets
  Bucket* buckets;
  gpuErrchk(cudaMallocManaged(&buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket)));

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  auto cc_config = CCAlgConfiguration().batch_factor(6);

  CCGPUSketchAlg cc_gpu_alg{num_nodes, num_updates, num_threads, buckets, get_seed(), sketchParams, cc_config};
  GraphSketchDriver<CCGPUSketchAlg> driver{&cc_gpu_alg, &stream, driver_config, reader_threads};
  
  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);

  auto cc_start = std::chrono::steady_clock::now();
  driver.prep_query(CONNECTIVITY);
  cudaDeviceSynchronize();
  auto flush_end = std::chrono::steady_clock::now();
  
  // Prefetch sketches back to CPU
  gpuErrchk(cudaMemPrefetchAsync(buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket), cudaCpuDeviceId));

  auto CC_num = cc_gpu_alg.connected_components().size();

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> flush_time = flush_end - driver.flush_start;
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
