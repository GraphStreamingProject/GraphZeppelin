#include "ingest.h"

vector<set<Node>> ingest_con_comp (istream * updates_stream)
{
	int n, m;
        *updates_stream >> n >> m;

        Graph G{n};

        int t, u, v;
        UpdateType type;
        for (int i = 0; i < m; i++)
        {
                *updates_stream >> t >> u >> v;
                type = t ? DELETE : INSERT;

                G.update({{u, v}, type});
        }

        return G.connected_components();
}

bool ingest_boost_bip (istream * updates_stream)
{
	using namespace boost;
	// Most space efficient graph representation in BGL
	typedef adjacency_list<vecS, vecS, undirectedS> undir_graph;

	int n, m;
        *updates_stream >> n >> m;
    
        undir_graph g{n};

        int upd_type, u, v;
        for (int i = 0; i < m; i++)
        {
                *updates_stream >> upd_type >> u >> v;
                if (upd_type == INSERT)
			add_edge(u, v, g);
		else
			remove_edge(u, v, g);
        }
	
	return is_bipartite(g);
}
