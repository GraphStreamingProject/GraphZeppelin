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
  int ierr = MPI_Init(&argc, &argv);
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
	int tagA = 0;
	int tagB = 0;
	{
		 std::cout << "Worker receiving broadcast" << std::endl;
		 MPI_Status status;
                 MPI_Probe(0, tagA,MPI_COMM_WORLD,&status);
                 int number_amount = 0;
                 MPI_Get_count(&status, MPI_CHAR, &number_amount);
                 char bytestream[number_amount];
                 MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, tagA++, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
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
	      MPI_Probe(0, tagA,MPI_COMM_WORLD,&status);
	      int number_amount = 0;
	      MPI_Get_count(&status, MPI_CHAR, &number_amount);
	      std::cout << "Number amount: " << number_amount << std::endl;
	      char bytestream[number_amount];
	      //first receive the work
	      int ierr = MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, tagA++, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	      std::cout << "received: " << (int)bytestream[0] << std::endl;
	      data_ret_t batched_updates = WorkQueue::deserialize_data_ret_t(bytestream);
	      std::vector<vec_t> updates = Graph::make_updates(batched_updates.first,batched_updates.second);
	      uint64_t node = batched_updates.first;
	      
	      //do the work
	      Supernode supernode(num_nodes, seed);
	      supernode.batch_update(updates);
	      std::cout << "Seed of first sketch at worker: " << supernode.sketches[0].seed << std::endl;
	      //serialize the result
	      std::string serial_str;
              boost::iostreams::back_insert_device<std::string> inserter(serial_str);
	      boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
	      boost::archive::binary_oarchive oa(s);
	    
	      oa << node;
	      for (int i = 0; i < supernode.sketches.size(); i++){
		oa << supernode.sketches[i];
	      }
	      std::cout << "Node: " << node << std::endl;
	      s.flush();
	      std::cout << "Sketch seed before serialization: " << supernode.sketches[7].seed << std::endl;
	      std::cout << "Size of the serialized sketch: " << serial_str.length() << std::endl;
	      //for (int i = 0; i < serial_str.length(); i++){
 		// std::cout << (int)serial_str[i] << "\t";
	      //} 
	      //send serialized result to master
	      std::cout << "Sending back to master" << std::endl;
	      MPI_Ssend(&serial_str[0],serial_str.length(), MPI_CHAR, 0, tagB++, MPI_COMM_WORLD);
	}}
  ierr = MPI_Finalize();
  return result;
  }


