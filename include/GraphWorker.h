#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <queue>
#include <pthread.h>

// forward declarations
class TokuInterface;
class Graph;


class GraphWorker {
public:
	GraphWorker(int _id, Graph *_graph, TokuInterface *_db);
	~GraphWorker();

	static std::atomic_flag queue_lock;
	static std::queue<uint64_t> work_queue;
	static void startWorkers(Graph *_graph, TokuInterface *_db);
	static void stopWorkers();
private:
	static void *startWorker(void *obj) {
		((GraphWorker *)obj)->doWork();
		return NULL;
	}

	void doWork();
	pthread_t thr;
	Graph *graph;
	TokuInterface *db;
	int id;
	bool shutdown = false;
	static int num_workers;
	static const char *config_file;
	static GraphWorker **workers;
};

// TODO:
//  * lock the nodes: So that there are not multiple workers updating the
//    same sketch. Probably very unlikely (with big tau) but still good
//    to do.

#endif
