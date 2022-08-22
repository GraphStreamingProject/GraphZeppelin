#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cmath>

#include "../graph/api.h"
#include "../graph/versioned_graph.h"
#include "../algorithms/CC.h"

#include "BufferedEdgeInput.h"

int main (int argc, char * argv[])
{
	unsigned long buff_size;
	sscanf(argv[3], "%lu", &buff_size);

	BufferedEdgeInput buff_in{argv[1], buff_size};

	unsigned long num_nodes = buff_in.num_nodes;
	unsigned long num_updates = buff_in.num_edges;
	
	auto vg = empty_treeplus_graph();
	
	using namespace std::chrono;
	
	unsigned long update_batch_size;
	sscanf(argv[2], "%lu", &update_batch_size);
	typedef std::tuple<uintV, uintV> edge;
 	vector<edge> insert_buffer(2 * update_batch_size);
	vector<edge> delete_buffer(2 * update_batch_size);

	unsigned long insert_buffer_count = 0;
	unsigned long delete_buffer_count = 0;

	bool is_delete;
	edge e1, e2;
	std::tuple<uintV, uintV, uint8_t> update;
        unsigned long log_count = 0;
        ofstream query_file{argv[4]};

	unsigned long query_interval;
	sscanf(argv[5], "%lu", &query_interval);
        query_interval = round(num_updates / query_interval); 
        
	auto start_time = steady_clock::now();
	for (unsigned long i = 0; i < num_updates; i++)
	{
                if (log_count == query_interval)
                {			
			auto CC_start_time = steady_clock::now();
			auto s = vg.acquire_version();
			CC(s.graph);
			auto CC_end_time = steady_clock::now();
			vg.release_version(std::move(s));
			auto CC_time_secs = (duration<double, std::ratio<1, 1>>(
				CC_end_time - CC_start_time)).count();

			query_file << CC_time_secs << "\n";
			query_file.flush();

			log_count = 0;
		}
                else log_count++;

		buff_in.get_edge(update);
		e1 = std::make_tuple(std::get<0>(update), 
				std::get<1>(update));
		e2 = std::make_tuple(std::get<1>(update), 
				std::get<0>(update));

		is_delete = std::get<2>(update) == 1;

		if (!is_delete)
		{
			insert_buffer[insert_buffer_count++] = e1;
			insert_buffer[insert_buffer_count] = e2;
			
			if (insert_buffer_count == 2*update_batch_size-1)
			{
				std::sort(insert_buffer.begin(), 
						insert_buffer.end());
				vg.insert_edges_batch(2*update_batch_size,
						insert_buffer.data(), false, 
						true, num_nodes, false);
				insert_buffer_count = 0;
			}
			else
				insert_buffer_count++;
		}
		else 
		{
			delete_buffer[delete_buffer_count++] = e1;
			delete_buffer[delete_buffer_count] = e2;

			if (delete_buffer_count == 2*update_batch_size-1)
			{
				std::sort(delete_buffer.begin(), 
						delete_buffer.end());
				vg.delete_edges_batch(2*update_batch_size,
					       	delete_buffer.data(), false, 
						true, num_nodes, false);
				delete_buffer_count = 0;
			}
			else
				delete_buffer_count++;
		}
	}

	// Flush remainder of buffer contents
	std::sort(insert_buffer.begin(), 
			insert_buffer.begin() + insert_buffer_count);
	vg.insert_edges_batch(insert_buffer_count, insert_buffer.data(),
			false, true, num_nodes, false);
	std::sort(delete_buffer.begin(), 
			delete_buffer.begin() + insert_buffer_count);
	vg.delete_edges_batch(delete_buffer_count, delete_buffer.data(),
			false, true, num_nodes, false);
	

	auto CC_start_time = steady_clock::now();
	auto s = vg.acquire_version();
	CC(s.graph);
	auto CC_end_time = steady_clock::now();
	vg.release_version(std::move(s));
	auto CC_time_secs = (duration<double, std::ratio<1, 1>>(
			CC_end_time - CC_start_time)).count();	
	query_file << CC_time_secs << "\n";
	
	auto end_time = steady_clock::now();

	auto runtime_secs = (duration<double, std::ratio<1, 1>>(
		end_time - start_time)).count();
//	auto updates_per_second = num_updates / runtime_secs;
//		time_log_file << "Updates per second: " << updates_per_second
//		<< "\n";
//      	time_log_file << "Ingestion time: " << ingest_time_secs << "\n";
	std::cout << "Total runtime: " << runtime_secs << std::endl;
	
	return 0;
}
