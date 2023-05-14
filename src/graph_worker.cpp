#include "../include/graph_worker.h"
#include "../include/graph.h"
#include <cuda_graph.cuh>

#ifdef USE_FBT_F
#include <gutter_tree.h>
#endif

#include <string>
#include <iostream>

bool GraphWorker::shutdown = false;
bool GraphWorker::paused   = false; // controls whether threads should pause or resume work
bool GraphWorker::cudaEnabled = false;
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
long GraphWorker::supernode_size;
GraphWorker **GraphWorker::workers;
std::condition_variable GraphWorker::pause_condition;
std::mutex GraphWorker::pause_lock;

/***********************************************
 ******** GraphWorker Static Functions *********
 ***********************************************/
/* These functions are used by the rest of the
 * code to manipulate the GraphWorkers as a whole
 */
void GraphWorker::start_workers(Graph *_graph, GutteringSystem *_gts, long _supernode_size) {
  shutdown = false;
  paused   = false;
  supernode_size = _supernode_size;

  workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
  for (int i = 0; i < num_groups; i++) {
    workers[i] = new GraphWorker(i, _graph, _gts);
  }
}

void GraphWorker::start_workers(Graph *_graph, GutteringSystem *_gts, CudaGraph *_cudaGraph, long _supernode_size) {
  shutdown = false;
  paused   = false;
  supernode_size = _supernode_size;

  cudaEnabled = true;

  workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
  for (int i = 0; i < num_groups; i++) {
    workers[i] = new GraphWorker(i, _graph, _gts, _cudaGraph);
  }
}

void GraphWorker::stop_workers() {
  if (shutdown)
    return;

  shutdown = true;
  workers[0]->gts->set_non_block(true); // make the GraphWorkers bypass waiting in queue
  
  pause_condition.notify_all();      // tell any paused threads to continue and exit
  for (int i = 0; i < num_groups; i++) {
    delete workers[i];
  }
  free(workers);
}

void GraphWorker::pause_workers() {
  paused = true;
  workers[0]->gts->set_non_block(true); // make the GraphWorkers bypass waiting in queue

  // wait until all GraphWorkers are paused
  while (true) {
    std::unique_lock<std::mutex> lk(pause_lock);
    pause_condition.wait_for(lk, std::chrono::milliseconds(500), []{
      for (int i = 0; i < num_groups; i++)
        if (!workers[i]->get_thr_paused()) return false;
      return true;
    });
    

    // double check that we didn't get a spurious wake-up
    bool all_paused = true;
    for (int i = 0; i < num_groups; i++) {
      if (!workers[i]->get_thr_paused()) {
        all_paused = false; // a worker still working so don't stop
        break;
      }
    }
    lk.unlock();

    if (all_paused) return; // all workers are done so exit
  }
}

void GraphWorker::unpause_workers() {
  workers[0]->gts->set_non_block(false); // buffer-tree operations should block when necessary
  paused = false;
  pause_condition.notify_all();       // tell all paused workers to get back to work
  
  // wait until all GraphWorkers are unpaused
  while (true) {
    std::unique_lock<std::mutex> lk(pause_lock);
    pause_condition.wait_for(lk, std::chrono::milliseconds(500), []{
      for (int i = 0; i < num_groups; i++)
        if (workers[i]->get_thr_paused()) return false;
      return true;
    });
    

    // double check that we didn't get a spurious wake-up
    bool all_unpaused = true;
    for (int i = 0; i < num_groups; i++) {
      if (workers[i]->get_thr_paused()) {
        all_unpaused = false; // a worker still working so don't stop
        break;
      }
    }
    lk.unlock();

    if (all_unpaused) return; // all workers are done so exit
  }
}

/***********************************************
 ************** GraphWorker class **************
 ***********************************************/
GraphWorker::GraphWorker(int _id, Graph *_graph, GutteringSystem *_gts) :
 id(_id), graph(_graph), gts(_gts), thr(start_worker, this), thr_paused(false) {
  delta_node = (Supernode *) malloc(supernode_size);
}

GraphWorker::GraphWorker(int _id, Graph *_graph, GutteringSystem *_gts, CudaGraph *_cudaGraph) :
 id(_id), graph(_graph), gts(_gts), cudaGraph(_cudaGraph), thr(start_worker, this), thr_paused(false) {
  delta_node = (Supernode *) malloc(supernode_size);
}

GraphWorker::~GraphWorker() {
  // join the GraphWorker thread to reclaim resources
  thr.join();
  free(delta_node);
}

void GraphWorker::do_work() {
  WorkQueue::DataNode *data;
  while(true) {
    // call get_data which will handle waiting on the queue
    // and will enforce locking.
    bool valid = gts->get_data(data);

    if (valid) {
      const std::vector<update_batch> &batches = data->get_batches();
      for (auto &batch : batches) {
        if (batch.upd_vec.size() > 0) {
          if (cudaEnabled) {
            cudaGraph->batch_update(id, batch.node_idx, batch.upd_vec);
          }
          else {
            graph->batch_update(batch.node_idx, batch.upd_vec, delta_node);
          }
        }

      }

      gts->get_data_callback(data); // inform guttering system that we're done
    }
    else if(shutdown)
      return;
    else if (paused) {
      std::unique_lock<std::mutex> lk(pause_lock);
      thr_paused = true; // this thread is currently paused
      pause_condition.notify_all(); // notify pause_workers()

      // wait until we are unpaused
      pause_condition.wait(lk, []{return !paused || shutdown;});
      thr_paused = false; // no longer paused
      lk.unlock();
      pause_condition.notify_all(); // notify unpause_workers()
    }
  }
}
