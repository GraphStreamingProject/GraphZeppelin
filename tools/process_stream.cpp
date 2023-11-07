#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include <sys/resource.h> // for rusage

static bool shutdown = false;

class TwoEdgeConnect{
public:
    CCSketchAlg cc_alg_1;
    CCSketchAlg cc_alg_2;
    node_id_t num_nodes;

    explicit TwoEdgeConnect(node_id_t num_nodes,
                            const CCAlgConfiguration& config_1, const CCAlgConfiguration& config_2)
            : cc_alg_1(num_nodes, config_1), cc_alg_2(num_nodes, config_2) {

    }

    void update(GraphUpdate upd){

    }

};


static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void track_insertions(uint64_t total, GraphSketchDriver<CCSketchAlg> *driver,
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
    size_t reader_threads = std::atol(argv[3]);

    BinaryFileStream stream(stream_file);
    BinaryFileStream stream2(stream_file);
    node_id_t num_nodes = stream.vertices();
    size_t num_updates  = stream.edges();
    std::cout << "Processing stream: " << stream_file << std::endl;
    std::cout << "nodes       = " << num_nodes << std::endl;
    std::cout << "num_updates = " << num_updates << std::endl;
    std::cout << std::endl;

    auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
    auto cc_config = CCAlgConfiguration().batch_factor(1);
    auto driver_config2 = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
    auto cc_config2 = CCAlgConfiguration().batch_factor(1);

    TwoEdgeConnect two_edge_alg{num_nodes, cc_config, cc_config2};

    // CCSketchAlg cc_alg{num_nodes, cc_config};
    GraphSketchDriver<CCSketchAlg> driver{&two_edge_alg.cc_alg_1, &stream, driver_config, reader_threads};

    /************
     * The plan is to use the constructor as follows.
     * GraphSketchDriver<TwoEdgeConnect> driver{&two_edge_alg, &stream, driver_config, reader_threads};
     *
    **************/

    // CCSketchAlg cc_alg2{num_nodes, cc_config2};
    GraphSketchDriver<CCSketchAlg> driver2{&two_edge_alg.cc_alg_2, &stream2, driver_config2, reader_threads};

    auto ins_start = std::chrono::steady_clock::now();
    std::thread querier(track_insertions, num_updates, &driver, ins_start);

    driver.process_stream_until(END_OF_STREAM);
    driver2.process_stream_until(END_OF_STREAM);

    auto cc_start = std::chrono::steady_clock::now();
    driver.prep_query();
    driver2.prep_query();
    auto CC_num = two_edge_alg.cc_alg_1.connected_components().size();

    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest = two_edge_alg.cc_alg_1.calc_spanning_forest();

    GraphUpdate temp_edge;

    temp_edge.type = DELETE;

    for(unsigned int j=0;j<forest.size();j++)
    {
        std::cout <<"Node "<< j <<": " << std::endl;
        two_edge_alg.cc_alg_2.apply_update_batch(0,forest[j].first,forest[j].second);
        for(unsigned int i=0;i<forest[j].second.size();i++)
        {
            std::cout <<"Nbr "<< i <<": " << forest[j].second[i] << std::endl;
            temp_edge.edge.src=forest[j].first;
            temp_edge.edge.dst=forest[j].second[i];
            two_edge_alg.cc_alg_2.update(temp_edge);
        }
    }

    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest2 = two_edge_alg.cc_alg_2.calc_spanning_forest();

    for(unsigned int j=0;j<forest2.size();j++)
    {
        std::cout <<"Node "<< j <<": " << std::endl;
        for(unsigned int i=0;i<forest2[j].second.size();i++)
        {
            std::cout <<"Nbr "<< i <<": " << forest2[j].second[i] << std::endl;
        }
    }

    auto CC_num2 = two_edge_alg.cc_alg_2.connected_components().size();

    std::chrono::duration<double> insert_time = driver.flush_end - ins_start;
    std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
    std::chrono::duration<double> flush_time = driver.flush_end - driver.flush_start;
    std::chrono::duration<double> cc_alg_time = two_edge_alg.cc_alg_1.cc_alg_end - two_edge_alg.cc_alg_1.cc_alg_start;

    shutdown = true;
    querier.join();

    double num_seconds = insert_time.count();
    std::cout << "Total insertion time(sec):    " << num_seconds << std::endl;
    std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;
    std::cout << "Total CC query latency:       " << cc_time.count() << std::endl;
    std::cout << "  Flush Gutters(sec):           " << flush_time.count() << std::endl;
    std::cout << "  Boruvka's Algorithm(sec):     " << cc_alg_time.count() << std::endl;
    std::cout << "Connected Components:         " << CC_num << std::endl;
    std::cout << "Connected Components2:         " << CC_num2 << std::endl;
    std::cout << "Maximum Memory Usage(MiB):    " << get_max_mem_used() << std::endl;
}
