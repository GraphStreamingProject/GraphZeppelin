#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>

#include "sketch.h"

// forward declarations
class Graph;
class GutteringSystem;

class WorkerThread {
 public:
  /**
   * Create a WorkerThread object by setting metadata and spinning up a thread.
   * @param _id     the id of the new WorkerThread.
   * @param _graph  the graph which this WorkerThread will be updating.
   * @param _gts    the database data will be extracted from.
   */
  WorkerThread(int _id, Graph *_graph, GutteringSystem *_gts, std::condition_variable &cond,
               std::mutex &lk);
  ~WorkerThread();

  void pause() { do_pause = true; }
  void unpause() {
    do_pause = false;
    paused = false;
  }

  void stop() { shutdown = true; }

  bool check_paused() { return paused; }

 private:

  /**
   * This function is used by a new thread to capture the WorkerThread object
   * and begin running do_work.
   * @param obj the memory where we will store the WorkerThread obj.
   */
  static void *start_worker(void *obj) {
    ((WorkerThread *)obj)->do_work();
    return nullptr;
  }

  void do_work();  // function which runs the WorkerThread process
  int id;
  Graph *graph;
  GutteringSystem *gts;
  std::condition_variable &flush_condition;
  std::mutex &flush_lock;
  bool shutdown = false;
  bool do_pause = false;
  bool paused = false;

  // the sketch this WorkerThread will use for generating deltas
  Sketch delta_sketch;

  // The thread that performs the work
  std::thread thr;
};

class WorkerThreadGroup {
 private:
  // list of all WorkerThreads
  WorkerThread **workers;
  size_t num_workers;

  std::condition_variable flush_condition;
  std::mutex flush_lock;

  GutteringSystem *gts;

 public:
  WorkerThreadGroup(size_t num_workers, Graph *_graph, GutteringSystem *_gts);
  ~WorkerThreadGroup();

  void flush_workers();
  void resume_workers();
};
