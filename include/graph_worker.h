#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

#ifndef USE_FBT_T
#include "work_queue.h"
#endif

// forward declarations
#ifdef USE_FBT_T
class BufferTree;
#endif
class Graph;


class GraphWorker {
public:
#ifdef USE_FBT_F
	GraphWorker(int _id, Graph *_graph, BufferTree *_db);
#else
  GraphWorker(int _id, Graph *_graph);
#endif
	~GraphWorker();

#ifdef USE_FBT_F
	static void startWorkers(Graph *_graph, BufferTree *_db);
#else
	static void startWorkers(Graph *_graph);
#endif
	static void stopWorkers();
	static int get_num_groups() {return num_groups;}
	static int get_group_size() {return group_size;}
	static void set_config(int g, int s) { num_groups = g; group_size = s; }
private:
	static void *startWorker(void *obj) {
		((GraphWorker *)obj)->doWork();
		return NULL;
	}

	void doWork();
	int id;
	Graph *graph;
#ifdef USE_FBT_F
	BufferTree *bf;
#else
	WorkQueue *wq;
#endif
	std::thread thr;

	static bool shutdown;
	static int num_groups;
	static int group_size;
	static GraphWorker **workers;
};

#endif
