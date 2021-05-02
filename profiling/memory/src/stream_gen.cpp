#include "../../test/util/graph_gen.h"
#include <cstdlib>

//Command line wrapper for graph_gen

int main (int argc, char * argv[])
{
	int n = atoi(argv[1]);
	srand(time(NULL));
	generate_stream({n, 0.03, 0.5, 0, argv[2], "cumul.txt"});

	return 0;
}
