#pragma once
#include <cache_guttering.h>
#include <gutter_tree.h>
#include <standalone_gutters.h>

#include "driver_configuration.h"
#include "graph_stream.h"
#include "worker_thread_group.h"
#ifdef VERIFY_SAMPLES_F
#include "graph_verifier.h"
#endif

class DriverException : public std::exception {
 private:
  std::string err_msg;
 public:
  DriverException(std::string msg) : err_msg(msg) {}
  virtual const char* what() const throw() {
    return err_msg.c_str();
  }
};

/**
 * GraphSketchDriver class:
 * Driver for sketching algorithms on a single machine.
 * Templatized by the "top level" sketching algorithm to manage.
 *
 * Algorithms need to implement the following functions to be managed by the driver:
 *
 *    1) void allocate_worker_memory(size_t num_workers)
 *          For performance reasons it is often helpful for the algorithm to allocate some scratch
 *          space to be used by individual worker threads. This scratch memory is managed by the
 *          algorithm. For example, in the connected components algorithm, we allocate a delta
 *          sketch for each worker.
 *
 *    2) size_t get_desired_updates_per_batch()
 *          Return the number of updates the algorithm would like us to batch. This serves as the
 *          maximum number of updates in a batch. We only provide smaller batches during
 *          prep_query()
 *
 *    3) node_id_t get_num_vertices()
 *          Returns the number of vertices in the Graph or an appropriate upper bound.
 *
 *    4) void pre_insert(GraphUpdate upd, node_id_t thr_id)
 *          Called before each update is added to the guttering system for the purpose of eager
 *          query heuristics. This function must be thread-safe and fast executing. The algorithm
 *          may choose to make this function a no-op.
 *
 *    5) void apply_update_batch(size_t thr_id, node_id_t src_vertex, const std::vector<node_id_t>
 *                               &dst_vertices)
 *          Called by worker threads to apply a batch of updates destined for a single vertex. This
 *          function must be thread-safe.
 *
 *    6) bool has_cached_query(int query_type)
 *          Check if the algorithm already has a cached answer for a given query type. If so, the
 *          driver can skip flushing the updates and applying them in prep_query(). The query_type
 *          should be defined by the algorithm as an enum (see cc_sketch_alg.h) but is typed in this
 *          code as an integer to ensure compatability across algorithms.
 *
 *    7) void print_configuration()
 *          Print the configuration of the algorithm. The algorithm may choose to print the
 *          configurations of subalgorithms as well.
 *
 *    8) void set_verifier(std::unique_ptr<GraphVerifier> verifier);
 *          If VERIFIER_SAMPLES_F is defined, then the driver provides the algorithm with a
 *          verifier. The verifier encodes the graph state at the time of a query losslessly
 *          and should be used by the algorithm to check its query answer. This is only used for
 *          correctness testing, not for production code.
 */
template <class Alg>
class GraphSketchDriver {
 private:
  GutteringSystem *gts;
  Alg *sketching_alg;
  GraphStream *stream;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier *verifier;
  std::mutex verifier_mtx;
#endif

  WorkerThreadGroup<Alg> *worker_threads;

  size_t num_stream_threads;
  static constexpr size_t update_array_size = 4000;

  std::atomic<size_t> total_updates;
 public:
  GraphSketchDriver(Alg *sketching_alg, GraphStream *stream, DriverConfiguration config,
                    size_t num_stream_threads = 1)
      : sketching_alg(sketching_alg), stream(stream), num_stream_threads(num_stream_threads) {
    sketching_alg->allocate_worker_memory(config.get_worker_threads());
    // set the leaf size of the guttering system appropriately
    if (config.gutter_conf().get_gutter_bytes() == GutteringConfiguration::uninit_param) {
      config.gutter_conf().gutter_bytes(sketching_alg->get_desired_updates_per_batch() *
                                        sizeof(node_id_t));
    }

    std::cout << config << std::endl;
    // Create the guttering system
    if (config.get_gutter_sys() == GUTTERTREE)
      gts = new GutterTree(config.get_disk_dir() + "/", sketching_alg->get_num_vertices(),
                           config.get_worker_threads(), config.gutter_conf(), true);
    else if (config.get_gutter_sys() == STANDALONE)
      gts = new StandAloneGutters(sketching_alg->get_num_vertices(), config.get_worker_threads(),
                                  num_stream_threads, config.gutter_conf());
    else
      gts = new CacheGuttering(sketching_alg->get_num_vertices(), config.get_worker_threads(),
                               num_stream_threads, config.gutter_conf());

    worker_threads = new WorkerThreadGroup<Alg>(config.get_worker_threads(), this, gts);
    sketching_alg->print_configuration();

    if (num_stream_threads > 1 && !stream->get_update_is_thread_safe()) {
      std::cerr
          << "WARNING: stream get_update is not thread safe. Setting number of stream threads to 1"
          << std::endl;
      num_stream_threads = 1;
    }
#ifdef VERIFY_SAMPLES_F
    verifier = new GraphVerifier(sketching_alg->get_num_vertices());
#endif

    total_updates = 0;
    std::cout << std::endl;
  }

  ~GraphSketchDriver() {
    delete worker_threads;
    delete gts;
#ifdef VERIFY_SAMPLES_F
    delete verifier;
#endif
  }

  /**
   * Processes the stream until a given edge index, at which point the function returns
   * @param break_edge_idx  the breakpoint edge index. All updates up to but not including this
   *                        index are processed by this call.
   * @throws DriverException if we cannot set the requested breakpoint.
   */
  void process_stream_until(edge_id_t break_edge_idx) {
    if (!stream->set_break_point(break_edge_idx)) {
      DriverException("Could not correctly set breakpoint: " + std::to_string(break_edge_idx));
      exit(EXIT_FAILURE);
    }
    worker_threads->resume_workers();

    auto task = [&](int thr_id) {
      GraphStreamUpdate update_array[update_array_size];
#ifdef VERIFY_SAMPLES_F
      GraphVerifier local_verifier(sketching_alg->get_num_vertices());
#endif

      while (true) {
        size_t updates = stream->get_update_buffer(update_array, update_array_size);
        for (size_t i = 0; i < updates; i++) {
          GraphUpdate upd;
          upd.edge = update_array[i].edge;
          upd.type = static_cast<UpdateType>(update_array[i].type);
          if (upd.type == BREAKPOINT) {
            // reached the breakpoint. Update verifier if applicable and return
#ifdef VERIFY_SAMPLES_F
            std::lock_guard<std::mutex> lk(verifier_mtx);
            verifier->combine(local_verifier);
#endif
            return;
          }
          else {
            sketching_alg->pre_insert(upd, thr_id);
            Edge edge = upd.edge;
            gts->insert({edge.src, edge.dst}, thr_id);
            gts->insert({edge.dst, edge.src}, thr_id);
#ifdef VERIFY_SAMPLES_F
            local_verifier.edge_update(edge);
#endif
          }
        }
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_stream_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (size_t i = 0; i < num_stream_threads; i++) threads[i].join();

    // pass the verifier to the algorithm
#ifdef VERIFY_SAMPLES_F
    sketching_alg->set_verifier(std::make_unique<GraphVerifier>(*verifier));
#endif
  }

  void prep_query(int query_code) {
    if (sketching_alg->has_cached_query(query_code)) {
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

#ifdef VERIFY_SAMPLES_F
  /**
   * checks that the verifier we constructed in process_stream_until matches another verifier
   * @param expected  the ground truth verifier
   * @throws DriverException if the verifiers do not match
   */
  void check_verifier(const GraphVerifier &expected) {
    if (*verifier != expected) {
      throw DriverException("Mismatch between driver verifier and expected verifier");
    }
  }
#endif

  size_t get_total_updates() { return total_updates.load(); }

  // time hooks for experiments
  std::chrono::steady_clock::time_point flush_start;
  std::chrono::steady_clock::time_point flush_end;
};