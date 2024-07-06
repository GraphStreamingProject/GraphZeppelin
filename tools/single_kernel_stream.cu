#include <vector>
#include <graph_sketch_driver.h>
#include <sk_gpu_sketch_alg.h>
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
void track_insertions(uint64_t total, GraphSketchDriver<SKGPUSketchAlg> *driver,
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
  auto cc_config = CCAlgConfiguration().batch_factor(6);
  SKGPUSketchAlg sk_gpu_alg{num_nodes, num_updates * 2, num_threads, get_seed(), cc_config};
  GraphSketchDriver<SKGPUSketchAlg> driver{&sk_gpu_alg, &stream, driver_config, reader_threads};
  
  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);
  std::cout << "Flushing... Current batch_count: " << sk_gpu_alg.get_batch_count() << "\n";

  auto flush_start = std::chrono::steady_clock::now();
  driver.prep_query(KSPANNINGFORESTS);
  auto flush_end = std::chrono::steady_clock::now();

  shutdown = true;
  querier.join();

  // Perform all sketch updates in gpu
  auto sketch_start = std::chrono::steady_clock::now();
  sk_gpu_alg.launch_gpu_kernel();
  auto sketch_end = std::chrono::steady_clock::now();

  // Apply delta sketch
  auto delta_start = std::chrono::steady_clock::now();
  sk_gpu_alg.apply_delta_sketch();
  auto delta_end = std::chrono::steady_clock::now();

  // Get CC
  auto cc_start = std::chrono::steady_clock::now();
  auto CC_num = sk_gpu_alg.connected_components().size();
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> sketch_time = sketch_end - sketch_start;
  std::chrono::duration<double> delta_time = delta_end - delta_start;

  std::cout << "GTS insertion time(sec):    " << insert_time.count() << std::endl;
  std::cout << "  Flush Gutters(sec):           " << flush_time.count() << std::endl;
  std::cout << "GPU time (sec):    " << sketch_time.count() << std::endl;
  std::cout << "Delta sketch applying time (sec):    " << delta_time.count() << std::endl;
  std::cout << "Total CC query latency:       " << cc_time.count() << std::endl;
  std::cout << "Connected Components:         " << CC_num << std::endl;
  std::cout << "Maximum Memory Usage(MiB):    " << get_max_mem_used() << std::endl;
}
