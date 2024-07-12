#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>

#include "sketch.h"

// forward declarations
template<class Alg>
class GraphSketchDriver;
class GutteringSystem;

/**
 * This class manages a thread of execution for performing sketch updates
 */
template<class Alg>
class WorkerThread {
 public:
  /**
   * Create a WorkerThread object by setting metadata and spinning up a thread.
   * @param _id       the id of the new WorkerThread.
   * @param _driver   the sketch algorithm driver this WorkerThread works for.
   * @param _gts      Guttering system to pull batches of updates from.
   * @param cond      a reference to the condition variable that coordinates the workers
   * @param lk        a reference to the lock that coordinates the workers
   */
  WorkerThread(int _id, GraphSketchDriver<Alg> *_driver, GutteringSystem *_gts,
               std::condition_variable &cond, std::mutex &lk)
      : id(_id),
        driver(_driver),
        gts(_gts),
        flush_condition(cond),
        flush_lock(lk),
        thr(start_worker, this) {}
  ~WorkerThread() {
    // join the WorkerThread thread to reclaim resources
    thr.join();
  }

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

  // function which runs the WorkerThread process
  void do_work() {
    WorkQueue::DataNode *data;
    while (true) {
      // call get_data which will handle waiting on the queue
      // and will enforce locking.
      bool valid = gts->get_data(data);

      if (valid) {
        const std::vector<update_batch> &batches = data->get_batches();
        for (auto &batch : batches) {
          if (batch.upd_vec.size() > 0) driver->batch_callback(id, batch.node_idx, batch.upd_vec);
        }
        gts->get_data_callback(data);  // inform guttering system that we're done
      } else if (shutdown)
        return;
      else if (do_pause) {
        std::unique_lock<std::mutex> lk(flush_lock);
        paused = true;                 // this thread is currently paused
        flush_condition.notify_all();  // notify that we are paused

        // wait until flush finished
        flush_condition.wait(lk, [&] { return !do_pause || shutdown; });
        paused = false;  // no longer paused
        lk.unlock();
        flush_condition.notify_all();  // notify that we are unpaused
      }
    }
  }
  const int id;
  GraphSketchDriver<Alg> *driver;
  GutteringSystem *gts;
  std::condition_variable &flush_condition;
  std::mutex &flush_lock;
  bool shutdown = false;
  bool do_pause = false;
  bool paused = false;

  // The thread that performs the work
  std::thread thr;
};

/**
 * This class manages a group of worker threads. Allowing the driver to start/flush/stop them.
 */
template<class Alg>
class WorkerThreadGroup {
 private:
  // list of all WorkerThreads
  WorkerThread<Alg> **workers;
  size_t num_workers;
  GutteringSystem *gts;

  std::condition_variable flush_condition;
  std::mutex flush_lock;

 public:
  WorkerThreadGroup(size_t num_workers, GraphSketchDriver<Alg> *driver, GutteringSystem *gts)
      : num_workers(num_workers), gts(gts) {
    workers = new WorkerThread<Alg> *[num_workers];
    for (size_t i = 0; i < num_workers; i++) {
      workers[i] = new WorkerThread<Alg>(i, driver, gts, flush_condition, flush_lock);
    }
  }
  ~WorkerThreadGroup() {
    gts->set_non_block(true);  // make the WorkerThreads bypass waiting in queue

    for (size_t i = 0; i < num_workers; i++) workers[i]->stop();

    flush_condition.notify_all();  // tell any paused threads to continue and exit
    for (size_t i = 0; i < num_workers; i++) delete workers[i];
    delete[] workers;
  }

  void flush_workers() {
    gts->set_non_block(true);  // make the WorkerThreads bypass waiting in queue
    for (size_t i = 0; i < num_workers; i++) workers[i]->pause();

    // wait until all WorkerThreads are flushed
    while (true) {
      std::unique_lock<std::mutex> lk(flush_lock);
      flush_condition.wait_for(lk, std::chrono::milliseconds(500), [&] {
        for (size_t i = 0; i < num_workers; i++)
          if (!workers[i]->check_paused()) return false;
        return true;
      });

      // double check that we didn't get a spurious wake-up
      bool all_paused = true;
      for (size_t i = 0; i < num_workers; i++) {
        if (!workers[i]->check_paused()) {
          all_paused = false;  // a worker still working so don't stop
          break;
        }
      }
      lk.unlock();

      if (all_paused) break;  // all workers are done so exit
    }
  }
  void resume_workers() {
    // unpause the WorkerThreads
    for (size_t i = 0; i < num_workers; i++) workers[i]->unpause();

    gts->set_non_block(false); // make WorkerThreads wait on the queue
    flush_condition.notify_all();
  }
};
