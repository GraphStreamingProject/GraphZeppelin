#include <random>
#include <queue>
#include <chrono>
#include <iostream>
#include <algorithm>

const int one_p = 2;
const long long n = 1e7;
const long long OVF_BUF = 1e17;


// simulates queuing with geometric arrival and constant servicing
int main() {
  double p = 1.0/one_p;
  std::geometric_distribution<unsigned long> geo {p};
  std::uniform_int_distribution<unsigned long long> uniform {OVF_BUF};
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937_64 generator(seed);
  std::priority_queue<long long> pq;
  
  long long queue_size = 0;
  long long max_size = 0;

  for (int i=0; i<n; ++i) {
    auto num_inserts = geo(generator);
    for (int j = 0; j < num_inserts; ++j) {
      pq.push(uniform(generator));
    }
    unsigned t = std::min(pq.size(), (size_t) one_p);
    for (int j = 0; j < t; ++j) {
      pq.pop();
    }
    queue_size += pq.size();
    if (pq.size() > max_size) max_size = pq.size();
    if (queue_size > OVF_BUF) {
      std::cout << "Oops, overflowed buf at sim number " << i << std::endl;
      return 0;
    }
  }
  std::cout << "Average number of elements in the queue: "
    << ((double)queue_size / n) << std::endl;
  std::cout << "Max number of elements in the queue: " << max_size << std::endl;
}