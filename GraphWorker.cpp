#include "include/GraphWorker.h"
#include "include/graph.h"
#include "include/TokuInterface.h"

std::atomic_flag GraphWorker::queue_lock = ATOMIC_FLAG_INIT;
std::queue<uint64_t> GraphWorker::work_queue;

struct timespec quarter_sec{0, 250000000};

GraphWorker::GraphWorker(int _id, Graph *_graph, TokuInterface *_db) {
	printf("Creating thread %llu\n", _id);
	pthread_create(&thr, NULL, GraphWorker::startWorker, this);
	graph = _graph;
	db = _db;
	id = _id;
}

GraphWorker::~GraphWorker() {
	shutdown = true;
	pthread_join(thr, NULL);
	printf("thread %llu joined!\n", id);
}

void GraphWorker::doWork() {
	while(true) {
		bool did_work = false;
		while (queue_lock.test_and_set(std::memory_order_acquire))
			; // spin-lock on the queue

		if (work_queue.empty() == false) {
			uint64_t node = work_queue.front();
			work_queue.pop();
			queue_lock.clear(std::memory_order_release);  // unlock
			if (db->update_counts[node] > 0) {
				// printf("Worker %d handling updates for node %llu\n", id, node);
				graph->batch_update(node, db->getEdges(node));
				did_work = true;
			}
		}
		else queue_lock.clear(std::memory_order_release); // unlock

		if (shutdown && work_queue.empty()) {// if recieved shutdown and there's no more work to do
			printf("Thread %i done and exiting\n", id);
			return;
		}

		if (!did_work) // if no work was done then sleep
			nanosleep(&quarter_sec, NULL);
	}
}
