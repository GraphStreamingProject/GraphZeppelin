#include <gtest/gtest.h>
#include <fstream>
#include "../include/work_queue.h"

#include <experimental/random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>

TEST(WorkQueueTestSuite, TestSpeed) {
  const auto count = 100000000;
  const auto node_count = 10000;
  std::vector<std::pair<node_id_t, node_id_t>> pairs;
  for (size_t i = 0; i < count; ++i)
  {
    auto update = std::make_pair<node_id_t, node_id_t>(std::experimental::randint(0, node_count), std::experimental::randint(0, node_count));
    pairs.push_back(update);
  }

  std::cout << "Using " << pairs.size() * sizeof(std::pair<node_id_t, node_id_t>) / 1e6 << " MB of memory" << std::endl;

  std::mutex _mutex;
  std::condition_variable _cv;
  const size_t num_threads = 1;
  size_t num_threads_waiting = 0;
  WorkQueue wq(100000, node_count, num_threads * 2);
  auto run_test = [&](std::pair<size_t, size_t> bounds){		    
		    //  auto run_test = [&wq, node_count, num_threads, &num_threads_waiting, &_mutex, &_cv](const std::vector<std::pair<node_id_t, node_id_t>> &pairs, std::pair<size_t, size_t> bounds){		    
		    std::unique_lock<std::mutex> lock(_mutex);
		    ++num_threads_waiting;
		    const bool ready = num_threads_waiting == num_threads;
		    lock.unlock();

		    if (ready)
		    {
		      _cv.notify_all();
		    }
		    else
		    {
		      lock.lock();
		      _cv.wait(lock, [&num_threads_waiting, &num_threads]()
				     {
				       return num_threads_waiting == num_threads;
				     });
		      lock.unlock();
		    }
		    
		    auto now = std::chrono::system_clock::now();
		    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		    std::cout << "Starting test at " << now_c << std::endl;
		    auto sum = 0;
		    for (const auto &pair : pairs)
		    {
		      //const auto i = pair.first + pair.second;
		      //sum += i;
		      if (pair.first >= bounds.first && pair.first < bounds.second)
		      wq.insert(pair);
		    }
		    std::cout << sum << std::endl;
		    std::cout << "Processing time is " << std::chrono::duration<double>(std::chrono::system_clock::now() - now).count() << std::endl;
		  };


  std::cout << sizeof(run_test) << std::endl;
  std::vector<std::thread> threads;
  const size_t quanta = node_count / num_threads;
  for (size_t i = 0; i < num_threads; ++i)
  {
    const std::pair<size_t, size_t> bounds = (i == (num_threads - 1)) ? std::make_pair(i * quanta, (size_t)node_count) : std::make_pair(i * quanta, (i + 1) * quanta);
    std::cout << "Making thread " << i << " with bounds " << bounds.first << " " << bounds.second << std::endl;
    threads.emplace_back(run_test, bounds);
  }
  const auto start = std::chrono::system_clock::now();
  for (auto &thread : threads)
  {
    thread.join();
  }
  const auto end = std::chrono::system_clock::now();
  std::cout << std::fixed << "Edges / sec: " << static_cast<double>(count) / std::chrono::duration<double>(end - start).count() << std::endl;
}
