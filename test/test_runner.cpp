#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <iostream>
#include <buffer_tree.h>
#include <vector>
#include "../include/work_queue.h"
#include "../include/graph.h"
#include "../include/supernode.h"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <unistd.h>
#include <sched.h>
#include "../include/graph_worker.h"

int main(int argc, char** argv) {
  int provided = -1;
  int ierr = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  if (provided != MPI_THREAD_MULTIPLE){
	std::cout << "ERROR!" << std::endl;
	exit(1);
  }
  int procid, P;
  int num_sketches = 10;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&P);
  std::cout << "Process with id: " << procid << " has this many allowed threads: " << omp_get_max_threads() << std::endl;
  testing::InitGoogleTest(&argc, argv);
  int result = 0;
  //master node runs tests
  if (procid == 0){
	//std::cout << "In master node" << std::endl;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(3, &mask);
	int rc = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
	if (rc != 0) {
	  std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
	}
	result = RUN_ALL_TESTS();
  }
  //worker nodes wait for work and then compute
  else{
	uint64_t num_nodes = -1;  
	time_t seed = 0;
	size_t num_terminates = 0;
	std::chrono::milliseconds total_mpi_receive_seed{};
	std::chrono::milliseconds total_time{};
	std::chrono::milliseconds total_mpi_receive_work{};
        std::chrono::milliseconds total_mpi_send_results{};
        std::chrono::milliseconds total_mpi_receive_terminate{};
        std::chrono::milliseconds total_computation{};
	auto start = std::chrono::high_resolution_clock::now();
	auto thread_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < GraphWorker::factor; ++i)
	{
		 //std::cout << "Worker receiving broadcast" << std::endl;
		 MPI_Status status;
                 MPI_Probe(0,0,MPI_COMM_WORLD,&status);	
                 int number_amount = 0;
                 MPI_Get_count(&status, MPI_CHAR, &number_amount);
                 char bytestream[number_amount];
                 MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                 total_mpi_receive_seed += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
		 std::string serial_str(bytestream,number_amount);
                 boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
                 boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s2(device);
                 boost::archive::binary_iarchive ia(s2);
                 ia >> num_nodes;
		 ia >> seed;
		 //std::cout << "Number of nodes at worker: " << num_nodes << std::endl;
		 //std::cout << "Graph seed at worker: " << seed << std::endl; 
	}
	while(true){
	      start = std::chrono::high_resolution_clock::now();
	      MPI_Status status;
	      //std::cout << "Worker probing " << procid << std::endl;
	      MPI_Probe(0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
	      if (status.MPI_TAG == 1){
		num_terminates += 1;
		char bytestream[1];
		std::cout << "Worker with id " << procid << " receiving terminate signal from Master" << std::endl;
		MPI_Recv(&bytestream[0],1,MPI_CHAR,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		if (num_terminates == GraphWorker::factor)
		  {
			total_mpi_receive_terminate += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
                        std::cout << "Worker with id " << procid << " received terminate signal from MasterA, sending terminate signal to MasterB" << std::endl;
                        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - thread_start);
			std::cout << "Total time that this worker  " << procid << " spent alive: " <<  total_time.count() << std::endl;
			std::cout << "Total time that this worker  " << procid << " spent on MPI receive work: " <<  total_mpi_receive_work.count() << std::endl;
			std::cout << "Total time that this worker  " << procid << " spent on MPI sending back result: " <<  total_mpi_send_results.count() << std::endl;
			std::cout << "Total time that this worker  " << procid << " spent on MPI receiving terminate: " <<  total_mpi_receive_terminate.count() << std::endl;
			std::cout << "Total time that this worker  " << procid << " spent on MPI receiving seed: " <<  total_mpi_receive_seed.count() << std::endl;
			std::cout << "Total time that this worker  " << procid << " spent on computation: " <<  total_computation.count() << std::endl;
			break;
		  }
		else
		  {
		    continue;
		  }
              }
	      int number_amount = 0;
	      MPI_Get_count(&status, MPI_CHAR, &number_amount);
	      //std::cout << "Worker with id: " << procid << " received number amount: " << number_amount << std::endl;
	      char* bytestream = new char[number_amount];
	      //first receive the work
	      //std::cout << "Worker receiving work" << std::endl;
	      int ierr = MPI_Recv(bytestream, number_amount, MPI_CHAR, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	      total_mpi_receive_work += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
	      //std::cout << "Worker received work: " << std::endl;
	      
	      start = std::chrono::high_resolution_clock::now();
              std::string serial_str;
              boost::iostreams::back_insert_device<std::string> inserter(serial_str);
              boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
              boost::archive::binary_oarchive oa(s);
	      size_t offset = 0;
	      while (offset < number_amount){
		      //std::cout << "Offset: " << offset << " with worker id: " << procid << std::endl;
              	      //INSERT CODE HERE
		      data_ret_t batched_updates = WorkQueue::deserialize_data_ret_t(bytestream+offset);
              	      std::vector<vec_t> updates = Graph::make_updates(batched_updates.first,batched_updates.second);
              	      uint64_t node = batched_updates.first;
              	      //do the work
                      Supernode supernode(num_nodes, seed);
                      supernode.batch_update(updates);
                      //std::cout << "Seed of first sketch at worker: " << supernode.sketches[0].seed << std::endl;
                      //serialize the result
		      oa << node;
              	      for (int i = 0; i < supernode.sketches.size(); i++){
                      	oa << supernode.sketches[i];
              	      }
                      size_t length = *reinterpret_cast<const size_t*>(bytestream + offset + sizeof(Node));
		      //std::cout << "Worker with id: " << procid << " length: " << length << std::endl;
	      	      offset += (length+1)*sizeof(Node) + sizeof(size_t);
	      }	      
	      //std::cout << "Node: " << node << std::endl;
	      s.flush();
	      //std::cout << "Sketch seed before serialization: " << supernode.sketches[7].seed << std::endl;
	      //std::cout << "Size of the serialized sketch: " << serial_str.length() << std::endl;
	      //for (int i = 0; i < serial_str.length(); i++){
 		// std::cout << (int)serial_str[i] << "\t";
	      //}
	      total_computation += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
	      //send serialized result to master
	      //std::cout << "Worker about to Send back to master " << procid << std::endl;
	      //std::cout << "serial_str size: " << serial_str.size() << std::endl;
	      start = std::chrono::high_resolution_clock::now();
	      MPI_Send(serial_str.data(),serial_str.length(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	      total_mpi_send_results += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
	      //std::cout << "Worker sent back to master " << procid << std::endl;
	      delete[] bytestream;
	}
  	std::cout << "Worker with id: " << procid << " gracefully terminating" << std::endl;
  }
  ierr = MPI_Finalize();
  return result;
  }


