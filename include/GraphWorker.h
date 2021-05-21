#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

// forward declarations
class BufferTree;
class Graph;


class GraphWorker {
public:
	GraphWorker(int _id, Graph *_graph, BufferTree *_db);
	~GraphWorker();

	static void startWorkers(Graph *_graph, BufferTree *_db);
	static void stopWorkers();
	static int get_group_size() {return group_size;}
private:
	static void *startWorker(void *obj) {
		((GraphWorker *)obj)->doWork();
		return NULL;
	}

	void doWork();
	int id;
	Graph *graph;
	BufferTree *bf;
	std::thread thr;

	static bool shutdown;
	static int num_groups;
	static int group_size;
	static const char *config_file;
	static GraphWorker **workers;
};

#endif
