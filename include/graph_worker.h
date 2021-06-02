#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

// forward declarations
class BufferTree;
class Graph;

class GraphWorker {
public:
	GraphWorker(int _id, Graph *_graph, BufferTree *_db);
	~GraphWorker();

	bool get_thr_paused() {return thr_paused;}

	// manage threads
	static void start_workers(Graph *_graph, BufferTree *_db);
	static void stop_workers();
	static void pause_workers();
	static void unpause_workers();

	// manage configuration
	static int get_num_groups() {return num_groups;}
	static int get_group_size() {return group_size;}
	static void set_config(int g, int s) { num_groups = g; group_size = s; }
private:
	static void *start_worker(void *obj) {
		((GraphWorker *)obj)->do_work();
		return NULL;
	}

	void do_work();
	int id;
	Graph *graph;
	BufferTree *bf;
	std::thread thr;
	std::atomic<bool> thr_paused; // indicates if this individual thread is paused

	// thread status and status management
	static bool shutdown;
	static bool paused;
	static std::condition_variable pause_condition;
	static std::mutex pause_lock;

	// configuration
	static int num_groups;
	static int group_size;

	// list of all GraphWorkers
	static GraphWorker **workers;
};

#endif
