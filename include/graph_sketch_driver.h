
#include <cache_guttering.h>
#include <gutter_tree.h>
#include <standalone_gutters.h>

#include "graph_configuration.h"
#include "graph_stream.h"
#include "worker_thread_group.h"

/**
 * GraphSketchDriver class:
 * Driver for sketching algorithms on a single machine
 * templatized by the algorithm.
 * 
 * Algorithms need to implement the following functions to be managed by the driver
 *    1) get_desired_updates_per_batch()
 * 
 *    2) pre_insert(upd, thr_id)
 *    
 *    3) apply_update_batch(thr_id, src_vertex, dst_vertices)
 *    
 *    4) has_cached_query() 
 */
template <class Alg>
class GraphSketchDriver {
 private:
  GutteringSystem *gts;
  Alg *sketching_alg;
  GraphStream *stream;

  WorkerThreadGroup<Alg> *worker_threads;

  size_t num_stream_threads;
  static constexpr size_t update_array_size = 4000;

  std::atomic<size_t> total_updates;
  FRIEND_TEST(GraphTest, TestSupernodeRestoreAfterCCFailure);

 public:
  GraphSketchDriver(Alg *sketching_alg, GraphStream *stream, GraphConfiguration config,
                    size_t num_inserters = 1)
      : sketching_alg(sketching_alg), stream(stream), num_stream_threads(num_inserters) {
    std::cerr << "Creating GraphSketchDriver!" << std::endl;
    // set the leaf size of the guttering system appropriately
    if (config.gutter_conf().get_gutter_bytes() == GutteringConfiguration::uninit_param) {
      config.gutter_conf().gutter_bytes(sketching_alg->get_desired_updates_per_batch() *
                                        sizeof(node_id_t));
    }

    // Create the guttering system
    if (config.get_gutter_sys() == GUTTERTREE)
      gts = new GutterTree(config.get_disk_dir(), sketching_alg->get_num_vertices(),
                           config.get_worker_threads(), config.gutter_conf(), true);
    else if (config.get_gutter_sys() == STANDALONE)
      gts = new StandAloneGutters(sketching_alg->get_num_vertices(), config.get_worker_threads(),
                                  num_stream_threads, config.gutter_conf());
    else
      gts = new CacheGuttering(sketching_alg->get_num_vertices(), config.get_worker_threads(),
                               num_stream_threads, config.gutter_conf());

    worker_threads = new WorkerThreadGroup<Alg>(config.get_worker_threads(), this, gts);

    if (num_stream_threads > 1 && !stream->get_update_is_thread_safe()) {
      std::cerr << "WARNING: stream get_update is not thread safe. Setting num inserters to 1"
                << std::endl;
      num_stream_threads = 1;
    }

    total_updates = 0;
    std::cout << config << std::endl;  // print the graph configuration
  }

  ~GraphSketchDriver() {
    std::cerr << "Shutting down GraphSketchDriver" << std::endl;
    delete worker_threads;
    delete gts;
  }

  void process_stream_until(edge_id_t break_edge_idx) {
    if (!stream->set_break_point(break_edge_idx)) {
      std::cerr << "ERROR: COULD NOT CORRECTLY SET BREAKPOINT!" << std::endl;
      exit(EXIT_FAILURE);
    }
    worker_threads->resume_workers();

    auto task = [&](int thr_id) {
      GraphStreamUpdate update_array[update_array_size];

      while (true) {
        size_t updates = stream->get_update_buffer(update_array, update_array_size);
        for (size_t i = 0; i < updates; i++) {
          GraphUpdate upd;
          upd.edge = update_array[i].edge;
          upd.type = static_cast<UpdateType>(update_array[i].type);
          if (upd.type == BREAKPOINT) {
            return;
          }
          else {
            sketching_alg->pre_insert(upd, thr_id);
            Edge edge = upd.edge;
            gts->insert({edge.src, edge.dst}, thr_id);
            gts->insert({edge.dst, edge.src}, thr_id);
          }
        }
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_stream_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (size_t i = 0; i < num_stream_threads; i++) threads[i].join();

    std::cerr << "Exiting because of breakpoint after " << total_updates << " updates" << std::endl;
    std::cerr << "Original target was: " << break_edge_idx << std::endl;
  }

  void prep_query() {
    if (sketching_alg->has_cached_query()) {
      flush_start = flush_end = std::chrono::steady_clock::now();
      return;
    }

    flush_start = std::chrono::steady_clock::now();
    gts->force_flush();
    worker_threads->flush_workers();
    flush_end = std::chrono::steady_clock::now();
  }

  inline void batch_callback(int thr_id, node_id_t src_vertex,
                             const std::vector<node_id_t> &dst_vertices) {
    total_updates += dst_vertices.size();
    sketching_alg->apply_update_batch(thr_id, src_vertex, dst_vertices);
  }

  size_t get_total_updates() { return total_updates.load(); }

  // time hooks for experiments
  std::chrono::steady_clock::time_point flush_start;
  std::chrono::steady_clock::time_point flush_end;
};
