#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <unistd.h>
#include <cmath>

using namespace std;

// NOTE: Only assumes up to 2^32 nodes in graph

int main (int argc, char * argv [])
{
	int notification_frequency = 5000000;

	string static_graph_file_name = argv[1];
	double static_reinsertion_param = atof(argv[2]);
	unsigned long static_reinsertion_cap;
	sscanf(argv[3], "%lu", &static_reinsertion_cap);
	double general_insertion_param = atof(argv[4]);
	unsigned long general_insertion_cap;
	sscanf(argv[5], "%lu", &general_insertion_cap);
	unsigned long num_general_inserts;
	sscanf(argv[6], "%lu", &num_general_inserts);
	unsigned int num_iso_nodes;
	sscanf(argv[7], "%u", &num_iso_nodes);
	string stream_file_name = argv[8];

	ifstream static_graph_file{static_graph_file_name};
	unsigned int num_nodes;
        unsigned long num_edges;
	static_graph_file >> num_nodes >> num_edges;

	typedef std::pair<unsigned int, unsigned int> edge;
	vector<edge> updates;
	// TODO: reserve some appropriate space to minimize resizing

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);

	unsigned long total_num_updates = 0;	

	// Static graph geometric inserts/deletes	
	
	cout << "Constructing updates for edges in input graph..." << endl;

	geometric_distribution<unsigned long> static_reinsertions(static_reinsertion_param);
	unsigned long u, v;
	unsigned long num_reinserts, num_updates;
	for (unsigned long i = 0; i < num_edges; i++)
	{
		if (i % notification_frequency == 0 && i != 0)
			cout << i << " edges completed..." << endl; 

		static_graph_file >> u >> v;

		// Isolate the last num_iso_nodes nodes 
		if (u > num_nodes - num_iso_nodes - 1 || v > num_nodes - num_iso_nodes - 1)
			continue;

		num_reinserts = min(static_reinsertion_cap, static_reinsertions(generator));
		// Want edges to be contained in final state of graph
		num_updates = 2 * num_reinserts + 1;

		for (unsigned long j = 0; j < num_updates; j++)
			updates.push_back(std::make_pair(u, v));

		total_num_updates += num_updates;
	}	

	// General geometric inserts/deletes	
	
	cout << "Constructing updates for random possible edges..." << endl;

	geometric_distribution<unsigned long> general_insertions(general_insertion_param);
	uniform_int_distribution<unsigned int> random_node(0, num_nodes - 1);
	unsigned long num_inserts;
	for (unsigned long i = 0; i < num_general_inserts; i++)
	{
		if (i % notification_frequency == 0 && i != 0)
			cout << i << " edges completed..." << endl; 

		unsigned int rand_u = random_node(generator);
		unsigned int rand_v = random_node(generator);
		
		num_inserts = min(general_insertion_cap, general_insertions(generator));
		// These edges do not persist to the end of the stream
		num_updates = 2 * num_inserts;

		for (unsigned long j = 0; j < num_updates; j++)
			updates.push_back(std::make_pair(rand_u, rand_v));

		total_num_updates += num_updates;
	}

	// In-memory shuffle of updates
	
	cout << "Conducting in-memory shuffle of generated updates..." << endl;
	std::shuffle(updates.begin(), updates.end(), generator);

	// Output updates to stream file and add update types
	
	cout << "Writing stream to file..." << endl;
	ofstream stream_file_out{stream_file_name}; 
	stream_file_out << num_nodes << ' ' << total_num_updates << '\n';

//	for (unsigned long i = 0; i < total_num_updates; i++)
//		stream_file_out << updates[i].first << '\t' << updates[i].second << '\n';	      
	
	vector<bool> edge_present(num_nodes * (num_nodes - 1), false);

        unsigned long index;
        char upd_type;
        for (unsigned long i = 0; i < total_num_updates; i++)
        {
                index = updates[i].first * num_nodes + updates[i].second;
                upd_type = edge_present[index] ? '1' : '0';
                edge_present[index] = !edge_present[index];
 
    		stream_file_out << upd_type << '\t' << 
			updates[i].first << '\t' << 
			updates[i].second << '\n';
        }


	return 0;
}
