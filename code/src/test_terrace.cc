#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <unistd.h>

#include "../include/integerSort/blockRadixSort/blockRadixSort.h"
#include "../include/graph.h"
#include "../include/util.h"
#include "../include/Components.h"

int main (int argc, char * argv[])
{
	using namespace std::chrono;
	
	ifstream stream_file{argv[1]};

	unsigned long num_nodes, num_updates;
	stream_file >> num_nodes >> num_updates;
	
	Graph g(num_nodes);
	
	unsigned long update_buffer_size;
	sscanf(argv[2], "%lu", &update_buffer_size);
      	
//	auto perm = get_random_permutation(update_buffer_size);
	std::vector<uint32_t> ins_srcs(update_buffer_size);
      	std::vector<uint32_t> ins_dests(update_buffer_size);
	pair_uint *edges = 
		(pair_uint*)calloc(update_buffer_size, sizeof(pair_uint));
	
	uint32_t u, v;
	unsigned long insert_buffer_count = 0;
	bool is_delete;
	auto ingest_start_time = steady_clock::now();
	for (unsigned long i = 0; i < num_updates; i++)
	{
		stream_file >> is_delete >> u >> v;

		if (!is_delete)
		{
			edges[insert_buffer_count].x = u;
			edges[insert_buffer_count].y = v;

			if (insert_buffer_count == update_buffer_size-1)
			{
				integerSort_y((pair_els *)edges, 
						update_buffer_size,
						num_nodes);
				integerSort_x((pair_els *)edges, 
						update_buffer_size,
						num_nodes);
				
				for(int j = 0; j < update_buffer_size; j++)
				{
					ins_srcs[j] = edges[j].x;
					ins_dests[j] = edges[j].y;
				}

				auto perm = get_random_permutation(
						update_buffer_size);

				g.add_edge_batch(ins_srcs.data(), 
					ins_dests.data(), 
					update_buffer_size, perm);

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

	ofstream time_log_file{argv[3]};

	time_log_file << "Updates per second: " << updates_per_second
		<< "\n"; 
	time_log_file << "Total runtime: " 
		<< ingest_time_secs + CC_time_secs;

	return 0;
}
