#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include "../../test/util/graph_gen.h"
#include "../util/ingest.h"
#include "../../include/bipartite.h"

#include <chrono>
#include <utility>

// Boiler-plate for timing execution of arbitrary function taken from:
// https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template<typename F, typename... Args>
double funcTime(F func, Args&&... args){
    TimeVar t1=timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow()-t1);
}

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
	
	int num_algos = 3;

	int domain_size = (stop - start) / step;
	vector<vector<double>> data(domain_size, 
			vector<double>(num_algos + 1, 0.0));

	for (int i = 0; i < domain_size; i++)
	{
		int n = start + i * step; 
		data[i][0] = n;

		cout << "Profiling on " << n << " vertex graphs..." << endl;

		for (int k = 1; k < num_trials + 1; k++)
		{
			generate_stream({n, 0.03, 0.5, 0, 
				"stream.txt", "cum_graph.txt"});
			ifstream * updates_stream = new ifstream (
					"stream.txt");
			
			// CC Profile
			double run_time = funcTime(ingest_con_comp, 
					updates_stream);
			data[i][1] = ((k - 1) * data[i][1]
				+ run_time) / k;	
			updates_stream->clear();
			updates_stream->seekg(0);	
		
			// is_bipartite Profile
			run_time = funcTime(is_bipartite, 
					updates_stream);
			data[i][2] = ((k - 1) * data[i][2]
				+ run_time) / k;	
			updates_stream->clear();
			updates_stream->seekg(0);	

			// BGL Bipartite Profile
			run_time = funcTime(ingest_boost_bip, 
					updates_stream);
			data[i][3] = ((k - 1) * data[i][3]
				+ run_time) / k;	

			delete updates_stream;
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
