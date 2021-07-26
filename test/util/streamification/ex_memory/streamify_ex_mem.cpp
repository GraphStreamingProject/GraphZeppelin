#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <unistd.h>
#include <cmath>

using namespace std;

int main (int argc, char * argv [])
{
	int notification_frequency = 500000000;

	if (argc != 9) {
		printf("Incorrect number of arguments! Expected 8. See README\n");
		exit(1);
	}

	string static_graph_file_name = argv[1];
	double static_reinsertion_param = atof(argv[2]);
	unsigned long static_reinsertion_cap;
	sscanf(argv[3], "%lu", &static_reinsertion_cap);
	double general_insertion_param = atof(argv[4]);
	unsigned long general_insertion_cap;
	sscanf(argv[5], "%lu", &general_insertion_cap);
	float general_inserts_factor;
	sscanf(argv[6], "%f", &general_inserts_factor);
	unsigned long num_iso_nodes;
	sscanf(argv[7], "%lu", &num_iso_nodes);
	string stream_file_name = argv[8];
	
	ifstream static_graph_file{static_graph_file_name};
	
	unsigned long num_nodes, num_edges;
	static_graph_file >> num_nodes >> num_edges;

	unsigned long num_general_inserts = num_edges * general_inserts_factor;
	std::cout << "num_nodes = " << num_nodes << ", stream_size = " << num_general_inserts << "\n";
	
	ofstream stream_file_out{stream_file_name}; 

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_int_distribution<int> rand_char(33, 126);

	int rand_prefix_len = log((double) num_edges * (2 * 1 / static_reinsertion_param + 1) 
			+ num_general_inserts * (2 * 1 / general_insertion_param)) / log(10.0);

	unsigned long total_num_updates = 0;	
	char * random_prefix = new char [rand_prefix_len];

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
		
		num_reinserts = min(static_reinsertion_cap, static_reinsertions(generator));

		if (u > num_nodes - num_iso_nodes - 1 || v > num_nodes - num_iso_nodes - 1) {
			// Isolate the last num_iso_nodes nodes so insert multiple of 2 times
			num_updates = 2 * num_reinserts + 2;
		} else {
			// Want edges to be contained in final state of graph
			num_updates = 2 * num_reinserts + 1;
		}

		for (unsigned long j = 0; j < num_updates; j++)
		{	
			for (int k = 0; k < rand_prefix_len; k++)
				random_prefix[k] = (char) rand_char(generator);
		
			stream_file_out << random_prefix << "\t" << u << "\t" << v << "\n";
		}

		total_num_updates += num_updates;
	}	

	// General geometric inserts/deletes	
	
	cout << "Constructing updates for random possible edges..." << endl;

	geometric_distribution<unsigned long> general_insertions(general_insertion_param);
	uniform_int_distribution<unsigned long> random_node(0, num_nodes - 1);
	unsigned long num_inserts;
	for (unsigned long i = 0; i < num_general_inserts; i++)
	{
		if (i % notification_frequency == 0 && i != 0)
			cout << i << " edges completed..." << endl; 

		unsigned long rand_u = random_node(generator);
		unsigned long rand_v = random_node(generator);
	
		// Avoid self-loops
                if (rand_u == rand_v)
                        continue;

		num_inserts = min(general_insertion_cap, general_insertions(generator));
		// These edges do not persist to the end of the stream
		num_updates = 2 * num_inserts;

		for (unsigned long j = 0; j < num_updates; j++)
		{	
			for (int k = 0; k < rand_prefix_len; k++)
				random_prefix[k] = (char) rand_char(generator);
		
			stream_file_out << random_prefix << "\t" << rand_u << "\t" << rand_v << "\n";
		}

		total_num_updates += num_updates;
	}

	delete [] random_prefix;
	stream_file_out.flush();

	// External memory sort by the random prefix to produce 
	// a random permutation

	cout << "Conducting random external memory permutation of updates..." << endl;
	system((string("sort -k 1,1 -S 75% --parallel=46 -t\'	\' -s -T /mnt/nvme/tmp ") + stream_file_name + string(" -o ") + stream_file_name).c_str());

	// Remove random prefixes used for permuting
	// Insert prefix denoting insertion or deletion
	
	cout << "Pruning temporary prefixes and adding update types..." << endl;

	vector<bool> edge_present(num_nodes * (num_nodes - 1), false);

	stream_file_out.clear();
	stream_file_out.seekp(0);
	// Mantain separate streams to avoid unnecessary flushes
	ifstream stream_file_in{stream_file_name};

	string prefix;
	unsigned long index;
	char upd_type;

	unsigned long temp_u, temp_v;
	stream_file_in >> prefix >> temp_u >> temp_v;
	stream_file_out << num_nodes << '\t' << total_num_updates << '\n';

	for (unsigned long i = 0; i < total_num_updates - 1; i++)
	{
		if (i % notification_frequency == 0 && i != 0)
			cout << i << " updates completed..." << endl;

		stream_file_in >> prefix >> u >> v;
	
		// NOTE: If (u, v) appears in addition to (v, u) in the
		// stream, there could be correctness issues here.

		index = u * num_nodes + v;
		upd_type = edge_present[index] ? '1' : '0';
		edge_present[index] = !edge_present[index];

		stream_file_out << upd_type << '\t' << u << '\t' << v << '\n';
	}

	// Reinsert femporarily removed first update
	index = temp_u * num_nodes + temp_v;
	upd_type = edge_present[index] ? '1' : '0';
	edge_present[index] = !edge_present[index];

	stream_file_out << upd_type << '\t' << temp_u << '\t' << temp_v << '\n';

	// Truncate remainder of file
	
	if (-1 == truncate(stream_file_name.c_str(), stream_file_out.tellp()))
	{
		cout << "Truncation error" << endl;
		return 1;
	}

	return 0;
}
