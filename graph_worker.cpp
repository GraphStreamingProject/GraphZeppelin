#include "include/graph_worker.h"
#include "include/graph.h"
#include <buffer_tree.h>

#include <fstream>
#include <string>

bool GraphWorker::shutdown = false;
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
GraphWorker **GraphWorker::workers;

void GraphWorker::startWorkers(Graph *_graph, BufferTree *_bf) {
	shutdown = false;

	workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
	for (int i = 0; i < num_groups; i++) {
		workers[i] = new GraphWorker(i, _graph, _bf);
	}
}

void GraphWorker::stopWorkers() {
	shutdown = true;
	for (int i = 0; i < num_groups; i++) {
		delete workers[i];
	}
	delete workers;
}

GraphWorker::GraphWorker(int _id, Graph *_graph, BufferTree *_bf) :
  id(_id), graph(_graph), bf(_bf), thr(startWorker, this) {
}

GraphWorker::~GraphWorker() {
	if(id == 0)
		bf->bypass_wait(); // to avoid race condition on shutdown. Tell threads to recheck wait
	
	// when a graph worker is deleted we must wait for the associated
	// thread to finish its work. See stopWorkers and doWork
	thr.join();
}

void GraphWorker::doWork() {
	data_ret_t data;
	while(true) {
		// call get_data which will handle waiting on the queue
		// and will enforce locking.
		bool valid = bf->get_data(data);

		if (valid)
			graph->batch_update(data.first, data.second);
		else if(shutdown)
			return;
	}
}
