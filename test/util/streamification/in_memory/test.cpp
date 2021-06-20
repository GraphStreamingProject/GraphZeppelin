#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>

int main(int argc, char * argv [])
{
	using namespace std;

	ifstream input_graph_file {argv[1]};
	unsigned int num_nodes = atoi(argv[2]);
	unsigned int update_frequency = atoi(argv[3]);

	vector<vector<bool>> edge_present(num_nodes, vector<bool>(num_nodes, false));

	unsigned int u, v;
	unsigned long count = 0;
	unsigned long k = 0;	
	while (input_graph_file >> u >> v)
	{
		if (count == update_frequency - 1)
		{
			std::cout << k + 1 << " edges read..." << std::endl;
			count = 0;
		}
		else count++;

		if (edge_present[v][u] == true)
		{
			std::cout << "Symmetric edge detected!" << std::endl;
			return 1;
		}

		edge_present[u][v] == true;

		k++;
	}
	

	return 0;
}
