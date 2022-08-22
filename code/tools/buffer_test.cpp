#include <iostream>
#include "BufferedEdgeInput.h"
#include <fstream>
#include <stdlib.h>

int main(int argc, char * argv[])
{
	BufferedEdgeInput buff_in{argv[1], atoi(argv[2])};

	std::tuple<unsigned int, unsigned int, bool> update;

	buff_in.get_edge(update);
	std::cout << std::get<0>(update) << " " << std::get<1>(update) 
		<< " " << std::get<2>(update) << std::endl;

	return 0;
}
