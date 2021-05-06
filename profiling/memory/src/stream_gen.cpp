#include "../../../test/util/graph_gen.h"
#include <cstdlib>

//Command line wrapper for graph_gen

int main (int argc, char * argv[])
{
	int n = atoi(argv[1]);
	srand(time(NULL));
	// Have p scale disprortionately to n to preserve mean 
	// edge-connectivity of about 10
	generate_stream({n, ((double) 1000 / n) * 0.04, 0.5, 0, argv[2], "cumul.txt"});

	return 0;
}
