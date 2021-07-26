#include <gtest/gtest.h>
#include <mpi.h>
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
  printf("procid: %d\n",procid);
  testing::InitGoogleTest(&argc, argv);
  int result = 0;
  //master node runs tests
  if (procid == 0){
	std::cout << "In master node" << std::endl;
	result = RUN_ALL_TESTS();
  }
  //worker nodes wait for work and then compute
  else{
	uint64_t num_nodes = -1;  
	time_t seed = 0;
	{
		 //std::cout << "Worker receiving broadcast" << std::endl;
		 MPI_Status status;
                 MPI_Probe(0,0,MPI_COMM_WORLD,&status);	
                 int number_amount = 0;
                 MPI_Get_count(&status, MPI_CHAR, &number_amount);
                 char bytestream[number_amount];
                 MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
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
	      MPI_Status status;
	      std::cout << "Worker probing " << procid << std::endl;
	      MPI_Probe(0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
	      if (status.MPI_TAG == 1){
                        char bytestream[1];
                        std::cout << "Worker with id " << procid << " receiving terminate signal from MasterA" << std::endl;
                        MPI_Recv(&bytestream[0],1,MPI_CHAR,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        std::cout << "Worker with id " << procid << " received terminate signal from MasterA, sending terminate signal to MasterB" << std::endl;
                        MPI_Ssend(&bytestream[0],1,MPI_CHAR,0,1,MPI_COMM_WORLD);
                        std::cout << "Worker with id " << procid << " successfuly sent terminate signal to MasterB. Now exiting" << std::endl;
                        break;
              }
	      int number_amount = 0;
	      MPI_Get_count(&status, MPI_CHAR, &number_amount);
	      //std::cout << "Number amount: " << number_amount << std::endl;
	      char bytestream[number_amount];
	      //first receive the work
	      std::cout << "Worker receiving work" << std::endl;
	      int ierr = MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	      std::cout << "Worker received work: " << std::endl;
	      data_ret_t batched_updates = WorkQueue::deserialize_data_ret_t(bytestream);
	      std::vector<vec_t> updates = Graph::make_updates(batched_updates.first,batched_updates.second);
	      uint64_t node = batched_updates.first;
	      
	      //do the work
	      Supernode supernode(num_nodes, seed);
	      supernode.batch_update(updates);
	      //std::cout << "Seed of first sketch at worker: " << supernode.sketches[0].seed << std::endl;
	      //serialize the result
	      std::string serial_str;
              boost::iostreams::back_insert_device<std::string> inserter(serial_str);
	      boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
	      boost::archive::binary_oarchive oa(s);
	      oa << node;
	      for (int i = 0; i < supernode.sketches.size(); i++){
		oa << supernode.sketches[i];
	      }
	      //std::cout << "Node: " << node << std::endl;
	      s.flush();
	      //std::cout << "Sketch seed before serialization: " << supernode.sketches[7].seed << std::endl;
	      //std::cout << "Size of the serialized sketch: " << serial_str.length() << std::endl;
	      //for (int i = 0; i < serial_str.length(); i++){
 		// std::cout << (int)serial_str[i] << "\t";
	      //}
	      std::string temp_string = serial_str; 
	      //send serialized result to master
	      std::cout << "Worker about to Send back to master " << procid << std::endl;
	      std::cout << "serial_str size: " << serial_str.size() << std::endl;
	      MPI_Ssend(temp_string.c_str(),temp_string.length(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	      std::cout << "Worker sent back to master " << procid << std::endl;
	}
  	std::cout << "Worker with id: " << procid << " gracefully terminating" << std::endl;
  }
  ierr = MPI_Finalize();
  return result;
  }


