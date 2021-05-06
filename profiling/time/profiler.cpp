#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "../../include/graph.h"
#include "../../test/util/graph_gen.h"

#include <chrono>
#include <utility>

typedef std::chrono::steady_clock::time_point time_var;

#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define time_now() std::chrono::steady_clock::now()

/* Command line input specify range of graph sizes (in terms of
 * number of vertices) to run on: start, stop, step
 * 	Note: stop is inclusive
 *
 * Additionally, the fourth command line arg indicates the number of
 * graphs of each size to evaluate
 */

using namespace std;

int main (int argc, char * argv[])
{
	int start = atoi(argv[1]);
	int stop = atoi(argv[2]);
	int step = atoi(argv[3]);
	int num_trials = atoi(argv[4]);
	
	int num_algos = 5;

	int domain_size = (stop - start) / step;
	vector<vector<double>> data(domain_size, 
			vector<double>(num_algos + 1, 0.0));

	for (int i = 0; i < domain_size; i++)
	{
		int n = start + i * step; 
		data[i][0] = n;

		cout << "Profiling on " << n << " vertex graph(s)..." 
			<< endl;

		for (int k = 1; k < num_trials + 1; k++)
		{
			cout << "Trial " << k << endl;

			srand(time(NULL));
			generate_stream({n, ((double) 1000 / n) * 0.04, 
				0.5, 0, 
				"stream.txt", "cum_graph.txt"});
			ifstream updates_stream{"stream.txt"};
		
		        int n, m;
		        updates_stream >> n >> m;
		
		        Graph g1{n};
		
		        int t, u, v;
		        UpdateType type;
		        for (int i = 0; i < m; i++)
		        {
		                updates_stream >> t >> u >> v;
                		type = t ? DELETE : INSERT;

                		g1.update({{u, v}, type});
        		}

			vector<Graph> graphs(num_algos - 1, g1);

			// CC Profile
			time_var t1= time_now();
			g1.connected_components();
			double run_time = duration(time_now()-t1);
			data[i][1] = ((k - 1) * data[i][1]
				+ run_time) / k;	
			updates_stream.clear();
			updates_stream.seekg(0);	
		
			// span_forest profile
			time_var t2= time_now();
			graphs[0].spanning_forest();
			run_time = duration(time_now()-t2);
			data[i][2] = ((k - 1) * data[i][2]
				+ run_time) / k;	
			updates_stream.clear();
			updates_stream.seekg(0);

			// k_edge_disjoint_span_forests_union profile
			time_var t3= time_now();
			graphs[1].k_edge_disjoint_span_forests_union(2);
			run_time = duration(time_now()-t3);
			data[i][3] = ((k - 1) * data[i][3]
				+ run_time) / k;	
			updates_stream.clear();
			updates_stream.seekg(0);

			time_var t4= time_now();
			graphs[2].k_edge_disjoint_span_forests_union(4);
			run_time = duration(time_now()-t4);
			data[i][4] = ((k - 1) * data[i][4]
				+ run_time) / k;	
			updates_stream.clear();
			updates_stream.seekg(0);

			time_var t5= time_now();
			graphs[3].k_edge_disjoint_span_forests_union(8);
			run_time = duration(time_now()-t5);
			data[i][5] = ((k - 1) * data[i][5]
				+ run_time) / k;	
		}
	}

	// Output data to file in markdown table format
	ofstream format_file{"time_log.txt"};
	for (auto& row : data)
	{
		format_file << "|";
		for (auto& elem : row)
			format_file << elem << "|";
		format_file << "\n";
	}

	return 0;
}
