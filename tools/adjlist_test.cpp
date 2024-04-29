#include <iostream>
#include <vector>
#include <thread>
#include <map>
#include <mutex>


struct AdjList {
  // Id: source vertex
  // Content: vector of dst vertices
  //std::vector<std::map<>> 
  std::map<int, std::map<int, int>> list;
  std::vector<std::mutex> src_mutexes;
};


int main(int argc, char** argv) {
    int num_nodes = 1;
    int num_neighbors = 5;
    int num_threads = 5;

    // Init adj. list
    AdjList adjlist;
    adjlist.src_mutexes = std::vector<std::mutex>(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        adjlist.list[i] = std::map<int, int>();
    }

    auto task = [&](int thr_id) {
        int src_vertex = 0;

        std::lock_guard<std::mutex> lk(adjlist.src_mutexes[src_vertex]);
        for (int i = 0; i < num_neighbors; i++) {
            int dst_vertex = ((thr_id / 2) * num_neighbors) + i;
            if (adjlist.list[src_vertex].find(dst_vertex) == adjlist.list[src_vertex].end()) {
                adjlist.list[src_vertex].insert({dst_vertex, 1});
            }
            else {
                adjlist.list[src_vertex].erase(dst_vertex);
            }
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (size_t i = 0; i < num_threads; i++) threads[i].join();

    // Print out adj.list
    for (int i = 0; i < num_nodes; i++) {
        std::cout << "Src: " << i << " Dst: ";
        for (auto j = adjlist.list[i].begin(); j != adjlist.list[i].end(); j++) {
            std::cout << j->first << " ";
        }
        std::cout << "\n";
    }
    return 0;
}