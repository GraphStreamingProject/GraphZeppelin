#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <pthread.h>

// forward declarations
class BufferTree;
class Graph;


class GraphWorker {
public:
	GraphWorker(int _id, Graph *_graph, BufferTree *_db);
	~GraphWorker();

	static void startWorkers(Graph *_graph, BufferTree *_db);
	static void stopWorkers();
private:
	static void *startWorker(void *obj) {
		((GraphWorker *)obj)->doWork();
		return NULL;
	}

	void doWork();
	pthread_t thr;
	Graph *graph;
	BufferTree *bf;
	int id;
	static bool shutdown;
	static int num_workers;
	static const char *config_file;
	static GraphWorker **workers;
};

// TODO:
//  * lock the nodes: So that there are not multiple workers updating the
//    same sketch. Probably very unlikely (with big tau) but still good
//    to do.

#endif
