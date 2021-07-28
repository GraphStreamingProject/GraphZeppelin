#include "include/graph_worker.h"
#include "include/graph.h"
#include <mpi.h>
#ifdef USE_FBT_F
#include <buffer_tree.h>
#endif
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>


bool GraphWorker::shutdown = false;
bool GraphWorker::paused   = false; // controls whether threads should pause or resume work
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
GraphWorker **GraphWorker::workers;
std::condition_variable GraphWorker::pause_condition;
std::mutex GraphWorker::pause_lock;

/***********************************************
 ******** GraphWorker Static Functions *********
 ***********************************************/
/* These functions are used by the rest of the
 * code to manipulate the GraphWorkers as a whole
 */

#ifdef USE_FBT_F
void GraphWorker::start_workers(Graph *_graph, BufferTree *_bf) {
	shutdown = false;
	paused   = false;
	workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
	for (int i = 0; i < num_groups; i++) {
		workers[i] = new GraphWorker(i, _graph, _bf);
	}
}
#else
void GraphWorker::start_workers(Graph *_graph, WorkQueue *_wq) {
  shutdown = false;
  paused   = false;
  int num_machines = 1;
  MPI_Comm_size(MPI_COMM_WORLD,&num_machines);
  //std::cout << "Number of machines found: " << num_machines << std::endl;
  set_config(num_machines-1,group_size);
  workers = (GraphWorker **) calloc(num_groups, sizeof(GraphWorker *));
  for (int i = 0; i < num_groups; i++) {
    workers[i] = new GraphWorker(i, _graph, _wq);
  }
}
#endif

void GraphWorker::stop_workers() {
	shutdown = true;
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#else
	workers[0]->wq->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#endif
	pause_condition.notify_all();      // tell any paused threads to continue and exit
	for (int i = 0; i < num_groups; i++) {
		delete workers[i];
	}
	delete workers;
}

void GraphWorker::pause_workers() {
	paused = true;
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#else
	workers[0]->wq->set_non_block(true); // make the GraphWorkers bypass waiting in queue
#endif

	// wait until all GraphWorkers are paused
	std::unique_lock<std::mutex> lk(pause_lock);
	pause_condition.wait(lk, []{
		for (int i = 0; i < num_groups; i++){
			if (!workers[i]->get_thr_paused()) return false;
		}
		return true;
			});
	lk.unlock();
}

void GraphWorker::unpause_workers() {
#ifdef USE_FBT_F
	workers[0]->bf->set_non_block(false); // buffer-tree operations should block when necessary
#else
  workers[0]->wq->set_non_block(false); // buffer-tree operations should block when necessary
#endif
	paused = false;
	pause_condition.notify_all();       // tell all paused workers to get back to work
}

/***********************************************
 ************** GraphWorker class **************
 ***********************************************/
#ifdef USE_FBT_F
GraphWorker::GraphWorker(int _id, Graph *_graph, BufferTree *_bf) :
  id(_id), graph(_graph), bf(_bf), thr(start_worker, this), thr_paused(false) {
}
#else
GraphWorker::GraphWorker(int _id, Graph *_graph, WorkQueue *_wq) :
      id(_id), graph(_graph), wq(_wq), thr(start_worker, this),
      thr_paused(false) {
}
#endif

GraphWorker::~GraphWorker() {
	// join the GraphWorker thread to reclaim resources
	thr.join();
}

void GraphWorker::do_work() {
	//std::cout << "Starting thread with id: " << id << std::endl;
	data_ret_t data;
	int count = 0;
        {	
		std::string serial_str;
		boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        	boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
      	        boost::archive::binary_oarchive oa(s);
       	 	oa << graph->num_nodes;
       	 	oa << graph->seed;
		s.flush();
		MPI_Ssend(&serial_str[0],serial_str.length(),MPI_CHAR,id+1,0,MPI_COMM_WORLD);
	}
	while(true){
#ifdef USE_FBT_F
		bool valid = bf->get_data(data);
#else
		bool valid = wq->get_data(data);
#endif
		if (valid){
			std::vector<char> bytestream_vector = WorkQueue::serialize_data_ret_t(data);
			char* bytestream = &bytestream_vector[0];
			int ierr = MPI_Send(bytestream, bytestream_vector.size(), MPI_CHAR, id+1, 0, MPI_COMM_WORLD);
		        int number_amount = 0;
			MPI_Status status;
                        //std::cout << "Probing worker for results: " << id+1 << std::endl;
                        MPI_Probe(id+1,0,MPI_COMM_WORLD,&status);
			MPI_Get_count(&status, MPI_CHAR, &number_amount);
			//std::cout << "Probe messaged size: " << number_amount << std::endl;
			char bytestream2[number_amount];
			//std::cout << "MasterB receiving" << std::endl;
			ierr = MPI_Recv(&bytestream2[0], number_amount, MPI_CHAR, id+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			//std::cout << "MasterB received" << std::endl;
			//deserialize the results into a vector of sketches
			std::string serial_str(bytestream2,number_amount);
			boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
			boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s2(device);
			boost::archive::binary_iarchive ia(s2);
			uint64_t node;
			ia >> node;
			Supernode supernode(graph->num_nodes, graph->seed);
			for (int i = 0; i < supernode.sketches.size(); i++){
				ia >> supernode.sketches[i];
			}
			//std::cout << "Node at master: " << node << std::endl;
			//std::cout << "Master received" << std::endl;
			//std::cout << "Sketch seed at master after serialization: " << supernode.sketches[7].seed << std::endl;
			//apply the sketch deltas to the graph
			graph->apply_supernode_deltas(node,supernode.sketches);
		}else if (paused){
			char* bytestream = "2";
			int ierr = MPI_Send(bytestream, 1, MPI_CHAR, id+1, 1, MPI_COMM_WORLD);
			std::lock_guard<std::mutex> lk(pause_lock);
			thr_paused = true;
			pause_condition.notify_all();
			//std::cout << "Thread designated to worker " << id+1 << " gracefully exiting." << std::endl;
			return;
	
		}
	}			
		
}
