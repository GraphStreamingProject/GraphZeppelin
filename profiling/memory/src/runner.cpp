#include "../../../include/graph.h"

int main (int argc, char * argv[])
{
	int algo_mode = atoi(argv[1]);
	ifstream updates_stream{argv[2]};

	int n, m;
        updates_stream >> n >> m;

        Graph G{n};

        int t, u, v;
        UpdateType type;
        for (int i = 0; i < m; i++)
        {   
                updates_stream >> t >> u >> v;
                type = t ? DELETE : INSERT;

                G.update({{u, v}, type});
        }   


	switch(algo_mode)
	{
		case 0:
			G.connected_components();
			break;
		case 1:
			G.spanning_forest();
			break;
		case 2:
			G.k_edge_disjoint_span_forests_union(3);
			break;
	}

	return 0;
}
