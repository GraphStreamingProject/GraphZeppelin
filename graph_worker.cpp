#include "include/graph_worker.h"
#include "include/graph.h"

#ifdef USE_FBT_F
#include <buffer_tree.h>
#endif

#include <fstream>
#include <string>

bool GraphWorker::shutdown = false;
bool GraphWorker::paused   = false; // controls whether threads should pause or resume work
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
GraphWorker **GraphWorker::workers;
std::condition_variable GraphWorker::pause_condition;
std::mutex GraphWorker::pause_lock;

/***********************************************
 ******** GraphWorker Static Functions *********
 ***********************************************/
/* These functions are used by the rest of the
 * code to manipulate the GraphWorkers as a whole
 */

#ifdef USE_FBT_F
void GraphWorker::start_workers(Graph *_graph, BufferTree *_bf) {
	shutdown = false;
	paused   = false;

	workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
	for (int i = 0; i < num_groups; i++) {
		workers[i] = new GraphWorker(i, _graph, _bf);
	}
}
#else
void GraphWorker::start_workers(Graph *_graph, WorkQueue *_wq) {
  shutdown = false;
  paused   = false;

  workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
  for (int i = 0; i < num_groups; i++) {
    workers[i] = new GraphWorker(i, _graph, _wq);
  }
}
#endif

void GraphWorker::stop_workers() {
	shutdown = true;
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#else
	workers[0]->wq->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#endif
	pause_condition.notify_all();      // tell any paused threads to continue and exit
	for (int i = 0; i < num_groups; i++) {
		delete workers[i];
	}
	delete workers;
}

void GraphWorker::pause_workers() {
	paused = true;
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#else
  workers[0]->wq->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#endif

	// wait until all GraphWorkers are paused
	std::unique_lock<std::mutex> lk(pause_lock);
	pause_condition.wait(lk, []{
		for (int i = 0; i < num_groups; i++)
			if (!workers[i]->get_thr_paused()) return false;
		return true;
	});
	lk.unlock();
}

void GraphWorker::unpause_workers() {
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(false); // buffer-tree operations should block when necessary
#else
  workers[0]->wq->set_non_block(false); // buffer-tree operations should block when necessary
#endif
	paused = false;
	pause_condition.notify_all();       // tell all paused workers to get back to work
}

/***********************************************
 ************** GraphWorker class **************
 ***********************************************/
#ifdef USE_FBT_F
GraphWorker::GraphWorker(int _id, Graph *_graph, BufferTree *_bf) :
  id(_id), graph(_graph), bf(_bf), thr(start_worker, this), thr_paused(false) {
}
#else
GraphWorker::GraphWorker(int _id, Graph *_graph, WorkQueue *_wq) :
      id(_id), graph(_graph), wq(_wq), thr(start_worker, this),
      thr_paused(false) {
}
#endif

GraphWorker::~GraphWorker() {
	// join the GraphWorker thread to reclaim resources
	thr.join();
}

void GraphWorker::do_work() {
	data_ret_t data;
	while(true) {
		if(shutdown)
			return;
		thr_paused = true; // this thread is currently paused
		pause_condition.notify_all(); // notify pause_workers()

		// wait until we are unpaused
		std::unique_lock<std::mutex> lk(pause_lock);
		pause_condition.wait(lk, []{return !paused || shutdown;});
		thr_paused = false; // no longer paused
		lk.unlock();
		while(!thr_paused) {
			// call get_data which will handle waiting on the queue
			// and will enforce locking.
#ifdef USE_FBT_F
			bool valid = bf->get_data(data);
#else
			bool valid = wq->get_data(data);
#endif

			if (valid)
				graph->batch_update(data.first, data.second);
			else if(shutdown)
				return;
			else if(paused)
				thr_paused = true; // pause this thread once no more updates
		}
	}
}
