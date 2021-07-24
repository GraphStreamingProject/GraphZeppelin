#include <gtest/gtest.h>
#include <mpi.h>
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
#

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
	//while(true){
	      MPI_Status status;
	      MPI_Probe(0, 0,MPI_COMM_WORLD,&status);
	      int number_amount = 0;
	      MPI_Get_count(&status, MPI_CHAR, &number_amount);
	      std::cout << "Number amount: " << number_amount << std::endl;
	      char bytestream[number_amount];
	      if (number_amount > 0){
	      	//first receive the work
		int ierr = MPI_Recv(&bytestream[0], number_amount, MPI_CHAR, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		std::cout << "received: " << (int)bytestream[0] << std::endl;
		data_ret_t batched_updates = WorkQueue::deserialize_data_ret_t(bytestream);
		std::vector<vec_t> updates = Graph::make_updates(batched_updates.first,batched_updates.second);
		
		//do the work
		Supernode supernode(10000, 403L);
		supernode.batch_update(updates);

		//send back
	      char buffer[10000];
	      boost::iostreams::basic_array_sink<char> sr(buffer, 10000);  
              boost::iostreams::stream< boost::iostreams::basic_array_sink<char> > source(sr);
              boost::archive::binary_oarchive oa(source);
              oa << supernode.sketches[0];
	      std::cout << "Sketch seed before serialization: " << supernode.sketches[0].seed << std::endl;
	      Sketch sketch;
	      boost::iostreams::basic_array_source<char> source2(buffer,10000);
	      boost::iostreams::stream<boost::iostreams::basic_array_source<char>> stream(source2);
	      boost::archive::binary_iarchive in_archive(stream);
              in_archive >> sketch;
              std::cout << "Sketch seed after serialization: " << sketch.seed << std::endl;
	      }else{
		std::cout << "Fatal error! number_amount is 0" << std::endl;
		exit(1);
	      }
		//data contains the batch updates, apply them and let result be the result
              //then send the result back to the master's other thread
	      //MPI_Send(result, result.size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	      //free(bytestream);
  }
  ierr = MPI_Finalize();
  return result;
  }


