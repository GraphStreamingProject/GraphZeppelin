#include "include/GraphWorker.h"
#include "include/graph.h"
#include "include/TokuInterface.h"

#include <fstream>
#include <string>

bool GraphWorker::shutdown = false;
std::mutex GraphWorker::queue_lock;
std::condition_variable GraphWorker::queue_cond;

std::queue<uint64_t> GraphWorker::work_queue;
int GraphWorker::num_workers = 1;
const char *GraphWorker::config_file = "graph_worker.conf";
GraphWorker **GraphWorker::workers;

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
	shutdown = true;
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
	pthread_join(thr, NULL);
}

void GraphWorker::doWork() {
	while(true) {
		std::unique_lock<std::mutex> queue_unique(queue_lock);
		queue_cond.wait(queue_unique, [this]{return (work_queue.empty() == false || shutdown);});

		if (work_queue.empty() == false) {
			uint64_t node = work_queue.front();
			work_queue.pop();
			queue_unique.unlock();  // unlock
			if (db->update_counts[node] > 0) {
				// printf("Worker %d handling updates for node %llu\n", id, node);
				graph->batch_update(node, db->getEdges(node));
			}
		}
		else queue_unique.unlock(); // unlock

		// doesn't really matter if this part isn't thread safe I believe
		// only reading and never should read empty when there is more to insert
		if (shutdown && work_queue.empty()) {// if recieved shutdown and there's no more work to do
			printf("Thread %i done and exiting\n", id);
			return;
		}
	}
}
