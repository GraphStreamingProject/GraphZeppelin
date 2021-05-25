#include "include/graph_worker.h"
#include "include/graph.h"
#include <buffer_tree.h>

#include <fstream>
#include <string>

bool GraphWorker::shutdown = false;
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
const char *GraphWorker::config_file = "graph_worker.conf";
GraphWorker **GraphWorker::workers;

void GraphWorker::startWorkers(Graph *_graph, BufferTree *_bf) {
	std::string line;
	std::ifstream conf(config_file);
	if (conf.is_open()) {
		while(getline(conf, line)) {
			if(line.substr(0, line.find('=')) == "num_groups") {
				num_groups = std::stoi(line.substr(line.find('=')+1));
				printf("Number of groups = %i\n", num_groups);
			}
			if(line.substr(0, line.find('=')) == "group_size") {
				group_size = std::stoi(line.substr(line.find('=')+1));
				printf("Size of groups = %i\n", group_size);
			}
		}
			
	} else {
		printf("WARNING: Could not open thread configuration file!\n");
	}
	workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
	for (int i = 0; i < num_groups; i++) {
		workers[i] = new GraphWorker(i, _graph, _bf);
	}

	shutdown = false;
}

void GraphWorker::stopWorkers() {
	shutdown = true;
	for (int i = 0; i < num_groups; i++) {
		delete workers[i];
	}
	delete workers;
}

GraphWorker::GraphWorker(int _id, Graph *_graph, BufferTree *_bf) :
  id(_id), graph(_graph), bf(_bf), thr(startWorker, this) {
}

GraphWorker::~GraphWorker() {
	thr.join();
}

void GraphWorker::doWork() {
	while(true) {
		std::unique_lock<std::mutex> queue_unique(bf->queue_lock);
		bf->queue_cond.wait(queue_unique, [this]{return (bf->work_queue.empty() == false || shutdown);});

		if (bf->work_queue.empty() == false) {
			work_t task = bf->work_queue.front();
			bf->work_queue.pop();
			queue_unique.unlock();  // unlock
			// printf("Worker %d handling updates for node %llu\n", id, node);
			std::pair<Node, std::vector<Node>> data = bf->get_data(task);
			graph->batch_update(data.first, data.second);
		}
		else queue_unique.unlock(); // unlock

		// doesn't really matter if this part isn't thread safe I believe
		// only reading and never should read empty when there is more to insert
		if (shutdown && bf->work_queue.empty()) {// if recieved shutdown and there's no more work to do
			printf("Thread %i done and exiting\n", id);
			return;
		}
	}
}
