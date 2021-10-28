#include "../include/graph_worker.h"
#include "../include/graph.h"

#ifdef USE_FBT_F
#include <gutter_tree.h>
#endif

#include <string>

bool GraphWorker::shutdown = false;
bool GraphWorker::paused   = false; // controls whether threads should pause or resume work
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
void GraphWorker::start_workers(Graph *_graph, BufferingSystem *_bf, long _supernode_size) {
  shutdown = false;
  paused   = false;
  supernode_size = _supernode_size;

  workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
  for (int i = 0; i < num_groups; i++) {
    workers[i] = new GraphWorker(i, _graph, _bf);
  }
}

void GraphWorker::stop_workers() {
  if (shutdown)
    return;

  shutdown = true;
  workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue
  
  pause_condition.notify_all();      // tell any paused threads to continue and exit
  for (int i = 0; i < num_groups; i++) {
    delete workers[i];
  }
  free(workers);
}

void GraphWorker::pause_workers() {
  paused = true;
  workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue

  // wait until all GraphWorkers are paused
  std::unique_lock<std::mutex> lk(pause_lock);
  pause_condition.wait_for(lk, std::chrono::milliseconds(500), []{
    for (int i = 0; i < num_groups; i++)
      if (!workers[i]->get_thr_paused()) return false;
    return true;
  });
  lk.unlock();
}

void GraphWorker::unpause_workers() {
  workers[0]->bf->set_non_block(false); // buffer-tree operations should block when necessary
  paused = false;
  pause_condition.notify_all();       // tell all paused workers to get back to work
}

/***********************************************
 ************** GraphWorker class **************
 ***********************************************/
GraphWorker::GraphWorker(int _id, Graph *_graph, BufferingSystem *_bf) :
 id(_id), graph(_graph), bf(_bf), thr(start_worker, this), thr_paused(false) {
  delta_node = (Supernode *) malloc(supernode_size);
}

GraphWorker::~GraphWorker() {
  // join the GraphWorker thread to reclaim resources
  thr.join();
  free(delta_node);
}

void GraphWorker::do_work() {
  data_ret_t data;
  while(true) {
    if(shutdown)
      return;
    std::unique_lock<std::mutex> lk(pause_lock);
    thr_paused = true; // this thread is currently paused
    lk.unlock();
    pause_condition.notify_all(); // notify pause_workers()

    // wait until we are unpaused
    lk.lock();
    pause_condition.wait(lk, []{return !paused || shutdown;});
    thr_paused = false; // no longer paused
    lk.unlock();
    while(true) {
      // call get_data which will handle waiting on the queue
      // and will enforce locking.
      if (paused) printf("waiting on data\n");
      bool valid = bf->get_data(data);
      if (paused) printf("got data valid = %s\n", valid? "true" : "false");

      if (valid)
        graph->batch_update(data.first, data.second, delta_node);
      else if(shutdown)
        return;
      else if(paused)
        break;
    }
  }
}
