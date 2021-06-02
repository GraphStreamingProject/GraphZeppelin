#include "include/graph_worker.h"
#include "include/graph.h"
#include <buffer_tree.h>

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

void GraphWorker::start_workers(Graph *_graph, BufferTree *_bf) {
	shutdown = false;

	workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
	for (int i = 0; i < num_groups; i++) {
		workers[i] = new GraphWorker(i, _graph, _bf);
	}
}

void GraphWorker::stop_workers() {
	shutdown = true;
	workers[0]->bf->bypass_wait(true); // make the GraphWorkers bypass waiting in queue
	pause_condition.notify_all();      // tell any paused threads to continue and exit
	for (int i = 0; i < num_groups; i++) {
		delete workers[i];
	}
	delete workers;
}

void GraphWorker::pause_workers() {
	paused = true;
	workers[0]->bf->bypass_wait(true); // make the GraphWorkers bypass waiting in queue

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
	workers[0]->bf->bypass_wait(false); // buffer-tree operations should block when necessary
	paused = false;
	pause_condition.notify_all();       // tell all paused workers to get back to work
}

/***********************************************
 ************** GraphWorker class **************
 ***********************************************/
GraphWorker::GraphWorker(int _id, Graph *_graph, BufferTree *_bf) :
  id(_id), graph(_graph), bf(_bf), thr(start_worker, this) {
}

GraphWorker::~GraphWorker() {
	// join the GraphWorker thread to reclaim resources
	thr.join();
}

void GraphWorker::do_work() {
	data_ret_t data;
	while(true) {
		if(shutdown)
			return;
		pause_lock.lock();
		thr_paused = true; // this thread is currently paused
		pause_lock.unlock();
		pause_condition.notify_one(); // notify pause_workers()

		// wait until we are unpaused
		std::unique_lock<std::mutex> lk(pause_lock);
		pause_condition.wait(lk, []{return !paused || shutdown;});
		thr_paused = false; // no longer paused
		lk.unlock();
		while(!thr_paused) {
			// call get_data which will handle waiting on the queue
			// and will enforce locking.
			bool valid = bf->get_data(data);

			if (valid)
				graph->batch_update(data.first, data.second);
			else if(shutdown)
				return;
			else if(paused)
				thr_paused = true; // pause this thread once no more updates
		}
	}
}
