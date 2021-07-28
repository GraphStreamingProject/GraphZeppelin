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
bool GraphWorker::all_workers_completed = false;
int GraphWorker::num_groups = 1;
int GraphWorker::group_size = 1;
int GraphWorker::next_worker = 1;
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
		return all_workers_completed;
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
	data_ret_t data;
	//std::cout << "doing work\n";
	int num_machines = 1;
	MPI_Comm_size(MPI_COMM_WORLD,&num_machines);
	int count = 0;
	if (id == 0) {
		//std::cout << "MasterA" << std::endl;
		{
			//std::cout << "broadcasting " << std::endl;
			std::string serial_str;
              		boost::iostreams::back_insert_device<std::string> inserter(serial_str);
              		boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
              		boost::archive::binary_oarchive oa(s);
			oa << graph->num_nodes;
		        oa << graph->seed;
			//std::cout << "Number of nodes at master: " << graph->num_nodes << std::endl;
		        //std::cout << "Graph seed at master: " << graph->seed << std::endl;
			//std::cout << "Seed of first sketch at master during broadcast " << graph->supernodes[0]->sketches[0].seed << std::endl; 
			s.flush();
			for (int i = 1; i < num_machines; i++){
				MPI_Ssend(&serial_str[0],serial_str.length(),MPI_CHAR,i,0,MPI_COMM_WORLD);
			}
		}
		while(true) {
			//while(!thr_paused) {
			// call get_data which will handle waiting on the queue
			// and will enforce locking.
#ifdef USE_FBT_F
				bool valid = bf->get_data(data);
#else
				bool valid = wq->get_data(data);
#endif
				//std::cout << "MasterA about to send data" << std::endl;
	
				if (valid){
		   			//std::cout << "Valid" << std::endl;			
					//send data to the next available worker
					std::vector<char> bytestream_vector = WorkQueue::serialize_data_ret_t(data);
					char* bytestream = &bytestream_vector[0];
					//std::vector<char> bytestream_vector;
					//bytestream_vector.push_back('e');
					//bytestream_vector.push_back('x');
					//char* bytestream = &bytestream_vector[0];
					//std::cout << "Size: " << bytestream_vector.size() << std::endl;
					//std::cout << "Data: " << (int)bytestream[0] << std::endl;
					//std::cout << "MasterA sending data " << std::endl;
					//std::cout << count++ << std::endl;
					int ierr = MPI_Send(bytestream, bytestream_vector.size(), MPI_CHAR, next_worker, 0, MPI_COMM_WORLD);
					//std::cout << "MasterA receiving data " << std::endl;
					//graph->batch_update(data.first, data.second);
					(next_worker += 1) %= num_machines;
					if (next_worker == 0){
					   next_worker++;
					}
				}else if (paused){
	     			   //std::cout << "Out of data" << std::endl;
				   char* bytestream = "2";
				   for (int i = 1; i < num_machines; i++){
					//std::cout << "MasterA sending STOP condition to worker with rank: " << i << std::endl;
					int ierr = MPI_Send(bytestream, 1, MPI_CHAR, i, 1, MPI_COMM_WORLD);
					//std::cout << "MasterA successfully sent STOP condition to worker with rank: " << i << std::endl;
				   }
				   //std::cout << "MasterA successfully sent STOP condition to all worker nodes." << std::endl;
				   return;
				}
			//}
		}
	}
	else{
		int completed_workers = 0;
		std:vector<bool> worker_complete(num_machines,false);
		worker_complete[0] = true; 
		for (int j = 1; j < num_machines; j++){
			if (worker_complete[j]){
				continue;
			}
			//std::cout << "MasterB" << std::endl;
			//receive the results from the worker
			MPI_Status status;
			//std::cout << "MasterB probing worker with rank: " << j << std::endl;
			MPI_Probe(j,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			//std::cout << "MasterB done probing worker with rank: " << j << std::endl;
			/*THE ABOVE 
			 *
			 */
			if (status.MPI_TAG == 1){
				//std::cout << "MasterB received stop probe from worker with id: " << j << std::endl;
				completed_workers++;
				worker_complete[j] = true;
				char bytestream[1];
				//std::cout << "MasterB trying to receive stop signal from worker with id: " << j << std::endl;
				MPI_Recv(&bytestream[0],1,MPI_CHAR,j,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				//std::cout << "MasterB received stop signal from worker with id: " << j << std::endl;
				if (completed_workers == num_machines - 1){
					//std::cout << "MasterB received stop signal from all workers!" << std::endl;
					std::lock_guard<std::mutex> lk(pause_lock);
					all_workers_completed = true;
					pause_condition.notify_all();		
					//std::cout << "MasterB gracefully terminating" << std::endl;
					return;
				}
			}else{
				int number_amount = 0;
				MPI_Get_count(&status, MPI_CHAR, &number_amount);
				//std::cout << "Probe messaged size: " << number_amount << std::endl;
				char bytestream[number_amount];
				//std::cout << "MasterB receiving" << std::endl;
				int ierr = MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, j, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				//std::cout << "MasterB received" << std::endl;
			
				//deserialize the results into a vector of sketches
				std::string serial_str(bytestream,number_amount);
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
                       	        //std::cout << "Done applying deltas" << std::endl;
			}	
			if (j == num_machines - 1){
					j = 0;
			}
		
		
		}	

	}
}
