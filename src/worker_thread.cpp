#include "../include/worker_thread.h"
#include "../include/graph.h"

#include <string>
#include <iostream>

/***********************************************
 ********* WorkerThreadGroup Functions *********
 ***********************************************/
/* These functions are used by the rest of the
 * code to manipulate the WorkerThread as a whole
 */
WorkerThreadGroup::WorkerThreadGroup(size_t num_workers, Graph *_graph, GutteringSystem *_gts)
    : num_workers(num_workers), gts(_gts) {
  workers = new WorkerThread *[num_workers];
  for (size_t i = 0; i < num_workers; i++) {
    workers[i] = new WorkerThread(i, _graph, gts, flush_condition, flush_lock);
  }
}

WorkerThreadGroup::~WorkerThreadGroup() {
  gts->set_non_block(true); // make the WorkerThread bypass waiting in queue

  for (size_t i = 0; i < num_workers; i++) {
    workers[i]->stop();
  }

  flush_condition.notify_all(); // tell any paused threads to continue and exit
  for (size_t i = 0; i < num_workers; i++) {
    delete workers[i];
  }
  delete[] workers;
}

void WorkerThreadGroup::flush_workers() {
  gts->set_non_block(true); // make the WorkerThread bypass waiting in queue
  for (size_t i = 0; i < num_workers; i++)
    workers[i]->pause();

  // wait until all WorkerThreads are flushed
  while (true) {
    std::unique_lock<std::mutex> lk(flush_lock);
    flush_condition.wait_for(lk, std::chrono::milliseconds(500), [&]{
      for (size_t i = 0; i < num_workers; i++)
        if (!workers[i]->check_paused()) return false;
      return true;
    });

    // double check that we didn't get a spurious wake-up
    bool all_paused = true;
    for (size_t i = 0; i < num_workers; i++) {
      if (!workers[i]->check_paused()) {
        all_paused = false; // a worker still working so don't stop
        break;
      }
    }
    lk.unlock();

    if (all_paused) break; // all workers are done so exit
  }
}

void WorkerThreadGroup::resume_workers() {
  // unpause the WorkerThreads
  for (size_t i = 0; i < num_workers; i++)
    workers[i]->unpause();

  flush_condition.notify_all();
}

/***********************************************
 ************** WorkerThread class **************
 ***********************************************/
WorkerThread::WorkerThread(int _id, Graph *_graph, GutteringSystem *_gts,
                           std::condition_variable &cond, std::mutex &lk)
    : id(_id),
      graph(_graph),
      gts(_gts),
      flush_condition(cond),
      flush_lock(lk),
      delta_sketch(graph->get_num_vertices(), graph->get_seed()),
      thr(start_worker, this) {}

WorkerThread::~WorkerThread() {
  // join the WorkerThread thread to reclaim resources
  thr.join();
}

void WorkerThread::do_work() {
  WorkQueue::DataNode *data;
  while(true) {
    // call get_data which will handle waiting on the queue
    // and will enforce locking.
    bool valid = gts->get_data(data);

    if (valid) {
      const std::vector<update_batch> &batches = data->get_batches();
      for (auto &batch : batches) {
        if (batch.upd_vec.size() > 0)
          graph->apply_batch_updates(batch.node_idx, batch.upd_vec, delta_sketch);
      }
      gts->get_data_callback(data); // inform guttering system that we're done
    }
    else if(shutdown)
      return;
    else if (do_pause) {
      std::unique_lock<std::mutex> lk(flush_lock);
      paused = true; // this thread is currently paused
      flush_condition.notify_all(); // notify that we are paused

      // wait until flush finished
      flush_condition.wait(lk, [&]{return !do_pause || shutdown;});
      paused = false; // no longer paused
      lk.unlock();
      flush_condition.notify_all(); // notify that we are unpaused
    }
  }
}
