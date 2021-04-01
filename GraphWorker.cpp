#include "include/GraphWorker.h"
#include "include/graph.h"
#include "include/TokuInterface.h"

#include <fstream>
#include <string>

std::atomic_flag GraphWorker::queue_lock = ATOMIC_FLAG_INIT;
std::queue<uint64_t> GraphWorker::work_queue;
int GraphWorker::num_workers = 1;
const char *GraphWorker::config_file = "graph_worker.conf";
GraphWorker **GraphWorker::workers;

struct timespec quarter_sec{0, 250000000};

void GraphWorker::startWorkers(Graph *_graph, TokuInterface *_db) {
	std::string line;
	std::ifstream conf(config_file);
	if (conf.is_open()) {
		getline(conf, line);
		printf("Thread configuration is %s\n", line.c_str());
		num_workers = std::stoi(line.substr(line.find('=')+1));
	}
	workers = (GraphWorker **) calloc(num_workers, sizeof(GraphWorker *));
	for (int i = 0; i < num_workers; i++) {
		workers[i] = new GraphWorker(i, _graph, _db);
	}
}

void GraphWorker::stopWorkers() {
	for (int i = 0; i < num_workers; i++) {
		delete workers[i];
	}
	delete workers;
}

GraphWorker::GraphWorker(int _id, Graph *_graph, TokuInterface *_db) {
	// printf("Creating thread %i\n", _id);
	pthread_create(&thr, NULL, GraphWorker::startWorker, this);
	graph = _graph;
	db = _db;
	id = _id;
}

GraphWorker::~GraphWorker() {
	shutdown = true;
	pthread_join(thr, NULL);
	// printf("thread %i joined!\n", id);
}

void GraphWorker::doWork() {
	while(true) {
		bool not_empty = false;
		while (queue_lock.test_and_set(std::memory_order_acquire))
			; // spin-lock on the queue

		if (work_queue.empty() == false) {
			uint64_t node = work_queue.front();
			work_queue.pop();
			queue_lock.clear(std::memory_order_release);  // unlock
			if (db->update_counts[node] > 0) {
				// printf("Worker %d handling updates for node %llu\n", id, node);
				graph->batch_update(node, db->getEdges(node));
			}
			not_empty = true;
		}
		else queue_lock.clear(std::memory_order_release); // unlock

		if (shutdown && work_queue.empty()) {// if recieved shutdown and there's no more work to do
			// printf("Thread %i done and exiting\n", id);
			return;
		}

		if (!not_empty) // if queue is empty than sleep for a bit
			nanosleep(&quarter_sec, NULL);
	}
}
