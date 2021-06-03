#ifndef WORKER_GUARD
#define WORKER_GUARD

#include <atomic>
#include <mutex>
#include <condition_variable>
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
  GraphWorker(int _id, Graph *_graph, WorkQueue *_wq);
#endif
	~GraphWorker();

	bool get_thr_paused() {return thr_paused;}

	// manage threads
#ifdef USE_FBT_F
  static void start_workers(Graph *_graph, BufferTree *_db);
#else
  static void start_workers(Graph *_graph, WorkQueue *_wq);
#endif
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
#ifdef USE_FBT_F
	BufferTree *bf;
#else
	WorkQueue *wq;
#endif
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
