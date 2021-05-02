#include "../../../include/bipartite.h"
#include "../../util/ingest.h"

int main (int argc, char * argv[])
{
	int algo_mode = atoi(argv[1]);
	ifstream * updates_stream = new ifstream(argv[2]);
	
	switch(algo_mode)
	{
		case 0:
			ingest_con_comp(updates_stream);
			break;
		case 1:
			is_bipartite(updates_stream);
			break;
		case 2:
			ingest_boost_bip(updates_stream);
			break;
	}
}
