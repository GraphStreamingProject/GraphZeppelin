#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>

#include "BufferedEdgeInput.h"
#include "../include/integerSort/blockRadixSort/blockRadixSort.h"
#include "../include/graph.h"
#include "../include/util.h"
#include "../include/Components.h"

int main (int argc, char * argv[])
{	
	if (argc != 5)
	{
		std::cout << "Incorrect number of arguments!" << std::endl;
		std::cout << "Arguments are: stream_file, batch_size, file_buffer_size, output_file" << std::endl;
		exit(EXIT_FAILURE);
	}
	unsigned long update_buffer_size;
	sscanf(argv[3], "%lu", &update_buffer_size);

	BufferedEdgeInput buff_in{argv[1], update_buffer_size};
		
	unsigned long num_nodes = buff_in.num_nodes;
	unsigned long num_updates = buff_in.num_edges;

	Graph g(num_nodes);

	unsigned long update_batch_size;
	sscanf(argv[2], "%lu", &update_batch_size);
      	
	std::vector<uint32_t> ins_srcs(update_batch_size);
      	std::vector<uint32_t> ins_dests(update_batch_size);
	pair_uint *edges = 
		(pair_uint*)calloc(update_batch_size, sizeof(pair_uint));

	using namespace std::chrono;

	uint32_t u, v;
	bool is_delete;
	std::tuple<uint32_t, uint32_t, uint8_t> update;
	unsigned long insert_buffer_count = 0;
	unsigned long log_count = 0;	
        ofstream time_log_file{argv[4]};
	unsigned long hundredth = round (num_updates / 100);
	double time_so_far_secs = 0;
	auto ingest_start_time = steady_clock::now();
	auto prev_log_time = steady_clock::now();
	for (unsigned long i = 0; i < num_updates; i++)
	{
		if (log_count == hundredth)
		{
			auto new_log_time = steady_clock::now();
			
			auto log_interval_secs = (duration<double, std::ratio<1, 1>>(new_log_time - prev_log_time)).count();
        		auto updates_per_second = log_count / log_interval_secs;
			time_so_far_secs += log_interval_secs;

			time_log_file << i / hundredth << "% :\n"; 
        		time_log_file << "Updates per second since last entry: " << updates_per_second << "\n";
        		time_log_file << "Time since last entry: " << log_interval_secs << "\n";
        		time_log_file << "Total runtime so far: " << time_so_far_secs << "\n\n";
			time_log_file.flush();
			log_count = 0;

			prev_log_time = steady_clock::now();
		}
		else log_count++;

		buff_in.get_edge(update);	
		is_delete = std::get<2>(update) == 1;
		u = std::get<0>(update);
		v = std::get<1>(update);

		if (!is_delete)
		{
			edges[insert_buffer_count].x = u;
			edges[insert_buffer_count].y = v;

			if (insert_buffer_count == update_batch_size-1)
			{

				auto perm = get_random_permutation(
						update_batch_size);
				
				integerSort_y((pair_els *)edges, 
						update_batch_size,
						num_nodes);
				integerSort_x((pair_els *)edges, 
						update_batch_size,
						num_nodes);
				
				for(int j = 0; j < update_batch_size; j++)
				{
					ins_srcs[j] = edges[j].x;
					ins_dests[j] = edges[j].y;
				}

				g.add_edge_batch(ins_srcs.data(), 
					ins_dests.data(), 
					update_batch_size, perm);

				insert_buffer_count = 0;
			}
			else
				insert_buffer_count++;
		}
		else 
		{
			g.remove_edge(u, v);
		}
	}

	// Flush remainder of buffer contents
	if (insert_buffer_count > 0)
	{
		auto perm = get_random_permutation(insert_buffer_count);
		g.add_edge_batch(ins_srcs.data(), ins_dests.data(), 
			insert_buffer_count, perm);
	}

	auto ingest_end_time = steady_clock::now();

	auto CC_start_time = steady_clock::now();
	CC(g);
	auto CC_end_time = steady_clock::now();

	free(edges);

	auto ingest_time_secs = (duration<double, std::ratio<1, 1>>(
		ingest_end_time - ingest_start_time)).count();
	auto updates_per_second = num_updates / ingest_time_secs;
	
	auto CC_time_secs = (duration<double, std::ratio<1, 1>>(
			CC_end_time - CC_start_time)).count();

	time_log_file << "Updates per second: " << updates_per_second
		<< "\n";
       	time_log_file << "Ingestion time: " << ingest_time_secs << "\n";	
	time_log_file << "Total runtime: " 
		<< ingest_time_secs + CC_time_secs;

	return 0;
}
